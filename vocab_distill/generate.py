"""Greedy decoding with the `tokenizers` backend, so new token IDs decode correctly."""
import torch


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens):
    """Greedy-decode a batch of rendered prompts (left-padded). Returns one dict per prompt."""
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    encoded = [tokenizer.encode(p, add_special_tokens=False).ids for p in prompts]
    width = max(map(len, encoded))
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(encoded):
        ids[i, width - len(row):] = torch.tensor(row, device=device)
        mask[i, width - len(row):] = 1
    position = (mask.cumsum(-1) - 1).clamp(min=0)
    # A single prompt needs no padding mask; the paper's DEV runs decoded this way.
    padded = len(prompts) > 1

    outputs = [[] for _ in prompts]
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    cache = None
    model.eval()
    for _ in range(max_new_tokens):
        extra = {'attention_mask': mask, 'position_ids': position} if padded else {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1, **extra)
        token = out.logits[:, -1].argmax(-1)
        for i, t in enumerate(token.tolist()):
            if not done[i]:
                outputs[i].append(t)
        done |= torch.tensor([t in eos for t in token.tolist()], device=device)
        if done.all():
            break
        cache = out.past_key_values
        ids = token.masked_fill(done, 0).unsqueeze(-1)
        mask = torch.cat([mask, (~done).long().unsqueeze(-1)], dim=-1)
        position = (mask.sum(-1) - 1).clamp(min=0).unsqueeze(-1)

    return [{'prediction': tokenizer.decode(out_ids, skip_special_tokens=True),
             'input_tokens': len(prompt_ids), 'output_tokens': len(out_ids),
             'hit_token_limit': not (out_ids and out_ids[-1] in eos) and len(out_ids) == max_new_tokens}
            for prompt_ids, out_ids in zip(encoded, outputs)]
