"""Generate predictions for an evaluation suite. Scoring happens in analyze.py.

Without --checkpoint this evaluates the unmodified model. The paper decoded DEV one
prompt at a time (--batch 1) and TEST in batches of 16; both are greedy.
"""
import argparse

from vocab_distill.common import chat_tokenizer, native_tokenizer, read_jsonl, render_prompt, write_jsonl
from vocab_distill.generate import generate
from vocab_distill.model import build_model, load_checkpoint, uses_new_tokens
from vocab_distill.vocab import build_tokenizer, decompose, load_vocab


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, help='e.g. data/eval/dev.jsonl.gz')
    p.add_argument('--out', required=True, help='predictions file, e.g. work/predictions/dev_distill.jsonl.gz')
    p.add_argument('--checkpoint')
    p.add_argument('--vocab', default='data/vocab256.json')
    p.add_argument('--batch', type=int, default=1)
    args = p.parse_args()

    native = native_tokenizer()
    merges = load_vocab(args.vocab)
    if args.checkpoint:
        model, method = load_checkpoint(args.checkpoint, decompose(merges))
    else:
        model, method = build_model('native'), 'native'
    tokenizer = build_tokenizer(native, merges) if uses_new_tokens(method) else native

    items = read_jsonl(args.suite)
    chat = chat_tokenizer()
    prompts = {r['id']: render_prompt(chat, r['prompt']) for r in items}
    results = {}
    # Group by output limit, and by prompt length so batches need little padding.
    for limit in sorted({r['max_new_tokens'] for r in items}):
        group = [r for r in items if r['max_new_tokens'] == limit]
        if args.batch > 1:
            group.sort(key=lambda r: len(prompts[r['id']]))
        for start in range(0, len(group), args.batch):
            chunk = group[start:start + args.batch]
            for row, out in zip(chunk, generate(model, tokenizer, [prompts[r['id']] for r in chunk], limit)):
                results[row['id']] = {**row, **out}
            print(f'{len(results)}/{len(items)}', flush=True)
    write_jsonl(args.out, [results[r['id']] for r in items])
    print(f'{method}: {len(items)} predictions -> {args.out}')


if __name__ == '__main__':
    main()
