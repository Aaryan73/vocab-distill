"""Answer every training prompt with the unmodified model (greedy). These answers are the training targets.

Raw-text rows are copied through unchanged. Answers that hit the length limit are kept
and marked `truncated`; training uses them as exact prefixes.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from vocab_distill.common import MODEL_ID, MODEL_REVISION, chat_tokenizer, read_jsonl, render_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts', default='work/prompts.jsonl')
    p.add_argument('--out', default='work/teacher.jsonl')
    p.add_argument('--batch', type=int, default=20)
    args = p.parse_args()

    rows = read_jsonl(args.prompts)
    chat = chat_tokenizer()
    chat.padding_side = 'left'
    stop = {chat.eos_token_id, chat.convert_tokens_to_ids('<|im_end|>')}
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').cuda().eval()

    # Batch prompts of similar length together; output order does not matter for training.
    todo = sorted((r for r in rows if r['kind'] == 'chat'), key=lambda r: (r['max_new_tokens'], len(r['prompt'])))
    with open(args.out, 'w') as f:
        for row in rows:
            if row['kind'] == 'raw':
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            inputs = chat([render_prompt(chat, r['prompt']) for r in batch], return_tensors='pt',
                          padding=True, add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max(r['max_new_tokens'] for r in batch),
                                        do_sample=False, temperature=None, top_p=None, top_k=None)
            for row, ids in zip(batch, output[:, inputs['input_ids'].shape[1]:].tolist()):
                answer = chat.decode(ids, skip_special_tokens=True).strip()
                truncated = not (stop & set(ids))
                f.write(json.dumps({**row, 'response': answer, 'truncated': truncated}, ensure_ascii=False) + '\n')
            print(f'{start + len(batch)}/{len(todo)}', flush=True)


if __name__ == '__main__':
    main()
