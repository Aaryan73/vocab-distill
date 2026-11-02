"""Train one method. Writes checkpoint.pt, stats.json and a per-step loss log to --out.

  distill      new rows + LoRA, on the teacher's answers, with boundary alignment (the paper's method)
  ce-only      same, cross-entropy only (ablation)
  naive        new rows + small adapter, next-token loss on raw text (baseline)
  native-lora  LoRA with the native tokenizer, next-token loss on raw text (baseline)

The paper's runs: distill and ce-only with --budget 1500000 (and distill with 3000000 for
the more-data ablation); the baselines with their default 1M-token budget.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch

from vocab_distill.common import SEED, chat_tokenizer, native_tokenizer, read_jsonl, write_json
from vocab_distill.losses import alignment_losses, next_token_loss
from vocab_distill.model import backbone, build_model, native_mode, save_checkpoint, uses_new_tokens
from vocab_distill.vocab import boundaries, build_tokenizer, decompose, load_vocab

LAYERS = [9, 18, 27]  # of 36; the final layer is always matched too


def eos_id(model):
    eos = model.generation_config.eos_token_id
    return eos[0] if isinstance(eos, list) else eos


def teacher_sequences(examples, native, student, parts, chat, eos, max_native):
    """Yield (native_ids, student_ids, ends, first_answer_token, task) for each teacher example."""
    for row in examples:
        if row['kind'] == 'raw':
            text, prompt = row['text'], None
        else:
            messages = [{'role': 'user', 'content': row['prompt']}]
            prompt = chat.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # A truncated answer is an exact prefix of what the teacher said, so no end-of-turn token.
            text = prompt + row['response'] if row['truncated'] else chat.apply_chat_template(
                messages + [{'role': 'assistant', 'content': row['response']}], tokenize=False)
        native_ids = native.encode(text, add_special_tokens=False).ids
        student_ids = student.encode(text, add_special_tokens=False).ids
        if row['kind'] == 'raw':
            native_ids, student_ids = native_ids + [eos], student_ids + [eos]

        # Cut long sequences at a boundary both tokenizations share.
        native_ids = native_ids[:max_native]
        keep, length = [], 0
        for token in student_ids:
            width = len(parts.get(token, [token]))
            if length + width > len(native_ids):
                break
            keep.append(token)
            length += width
        student_ids, native_ids = keep, native_ids[:length]
        try:
            ends = boundaries(native_ids, student_ids, parts)
        except ValueError:
            continue

        first = 0
        if prompt is not None:
            prompt_ids = student.encode(prompt, add_special_tokens=False).ids
            if student_ids[:len(prompt_ids)] != prompt_ids or len(student_ids) <= len(prompt_ids):
                continue  # prompt tokenized differently in context, or no answer left after the cut
            first = len(prompt_ids)
        yield native_ids, student_ids, ends, first, row['task']


def train_on_teacher(model, args, parts, native, student):
    """The paper's method (and its cross-entropy-only ablation)."""
    align = args.method == 'distill'
    examples = [r for r in read_jsonl(args.examples) if r['kind'] == 'raw' or r.get('response')]
    random.Random(SEED).shuffle(examples)
    rows = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith('new_rows')]
    lora = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith('new_rows')]
    optimizer = torch.optim.AdamW([{'params': rows, 'lr': 5e-4}, {'params': lora, 'lr': 1e-4}], weight_decay=0.01)
    head = model.get_output_embeddings()

    stats, log, start = Counter(), open(Path(args.out) / 'steps.jsonl', 'w'), time.monotonic()
    model.train()
    pending = 0
    sequences = teacher_sequences(examples, native, student, parts, chat_tokenizer(), eos_id(model), args.max_native)
    for native_ids, student_ids, ends, first, task in sequences:
        if stats['native_tokens'] + len(native_ids) > args.budget:
            break
        n = torch.tensor([native_ids], device='cuda')
        s = torch.tensor([student_ids], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = backbone(model)(input_ids=s, use_cache=False, output_hidden_states=align)
        hidden = kl = torch.zeros((), device='cuda')
        if align:
            model.eval()
            with torch.no_grad(), native_mode(model), torch.autocast('cuda', dtype=torch.bfloat16):
                teacher = backbone(model)(input_ids=n, use_cache=False, output_hidden_states=True)
            model.train()
            hidden, kl = alignment_losses(out, teacher, ends, LAYERS, head.original, args.kl_positions)
        # Cross-entropy on the answer only (from the last prompt token), or on all raw text.
        ce_start = max(first - 1, 0)
        ce = next_token_loss(head, out.last_hidden_state[:, ce_start:], s[:, ce_start:])
        loss = ce + hidden + kl
        (loss / args.accumulation).backward()
        pending += 1
        if pending == args.accumulation:
            torch.nn.utils.clip_grad_norm_(rows + lora, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            stats['updates'] += 1
        stats['examples'] += 1
        stats[task] += 1
        stats['native_tokens'] += len(native_ids)
        stats['student_tokens'] += len(student_ids)
        log.write(json.dumps({'example': stats['examples'], 'ce': ce.item(), 'hidden': hidden.item(),
                              'kl': kl.item(), 'seconds': round(time.monotonic() - start, 1)}) + '\n')
    if pending:
        torch.nn.utils.clip_grad_norm_(rows + lora, 1.0)
        optimizer.step()
        stats['updates'] += 1
    log.close()
    return stats


def text_segments(text, tokenizer, limit):
    """Split text into pieces of at most `limit` native tokens, preferring space or newline cuts."""
    while text:
        if len(tokenizer.encode(text, add_special_tokens=False).ids) <= limit:
            yield text
            return
        low, high = 1, len(text)
        while low < high:  # longest prefix that fits
            middle = (low + high + 1) // 2
            if len(tokenizer.encode(text[:middle], add_special_tokens=False).ids) <= limit:
                low = middle
            else:
                high = middle - 1
        cut = max(text[:low].rfind('\n'), text[:low].rfind(' '))
        cut = cut + 1 if cut > low // 2 else low
        yield text[:cut]
        text = text[cut:]


def train_on_raw_text(model, args, native, student):
    """The two baselines: plain next-token training on raw text, in file order."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    eos = eos_id(model)

    stats, log, start = Counter(), open(Path(args.out) / 'steps.jsonl', 'w'), time.monotonic()
    model.train()
    pending, done = 0, False
    for row in read_jsonl(args.raw_text):
        for text in text_segments(row['text'], native, 512):
            native_ids = native.encode(text, add_special_tokens=False).ids
            if stats['native_tokens'] + len(native_ids) > args.budget:
                done = True
                break
            # The budget counts native tokens of the text itself; EOS is added after.
            student_ids = student.encode(text, add_special_tokens=False).ids + [eos]
            s = torch.tensor([student_ids], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = backbone(model)(input_ids=s, use_cache=False).last_hidden_state
            loss = next_token_loss(model.get_output_embeddings(), hidden, s, chunk=8)
            (loss / args.accumulation).backward()
            pending += 1
            if pending == args.accumulation:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                stats['updates'] += 1
            stats['segments'] += 1
            stats['native_tokens'] += len(native_ids)
            stats['student_tokens'] += len(student_ids) - 1
            log.write(json.dumps({'segment': stats['segments'], 'loss': loss.item(),
                                  'seconds': round(time.monotonic() - start, 1)}) + '\n')
        if done:
            break
    if pending:  # last partial batch: undo the 1/accumulation scaling it did not fill
        for p in params:
            if p.grad is not None:
                p.grad.mul_(args.accumulation / pending)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        stats['updates'] += 1
    log.close()
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True, choices=['distill', 'ce-only', 'naive', 'native-lora'])
    p.add_argument('--out', required=True)
    p.add_argument('--vocab', default='data/vocab256.json')
    p.add_argument('--examples', default='data/train/teacher.jsonl.gz')
    p.add_argument('--raw-text', default='data/train/raw_text.jsonl.gz')
    p.add_argument('--budget', type=int, help='native tokens to train on (default 1.5M, or 1M for baselines)')
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--max-native', type=int, default=1024)
    p.add_argument('--kl-positions', type=int, default=128)
    args = p.parse_args()
    args.budget = args.budget or (1_500_000 if args.method in ('distill', 'ce-only') else 1_000_000)
    Path(args.out).mkdir(parents=True, exist_ok=False)

    native = native_tokenizer()
    merges = load_vocab(args.vocab)
    parts = decompose(merges)
    student = build_tokenizer(native, merges) if uses_new_tokens(args.method) else native
    model = build_model(args.method, parts)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    start = time.monotonic()
    if args.method in ('distill', 'ce-only'):
        stats = train_on_teacher(model, args, parts, native, student)
    else:
        stats = train_on_raw_text(model, args, native, student)
    stats = {**stats, 'minutes': round((time.monotonic() - start) / 60, 1),
             'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
             'peak_gpu_gib': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    save_checkpoint(Path(args.out) / 'checkpoint.pt', model, args.method, stats)
    write_json(Path(args.out) / 'stats.json', stats)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
