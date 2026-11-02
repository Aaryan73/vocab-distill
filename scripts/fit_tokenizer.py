"""Fit new Hindi merges on top of Qwen's tokenizer and check them.

Writes a small JSON file with the merges; vocab_distill.vocab.build_tokenizer turns it
back into a full tokenizer. The paper uses the first 256 merges (data/vocab256.json).
"""
import argparse
import json

from vocab_distill.common import native_tokenizer, read_jsonl, write_json
from vocab_distill.vocab import FIRST_NEW_ID, build_tokenizer, decompose, fit_merges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='work/data/tokenizer_hi.jsonl')
    p.add_argument('--dev', default='work/data/dev_hi.jsonl')
    p.add_argument('--merges', type=int, default=256)
    p.add_argument('--out', default='work/vocab256.json')
    args = p.parse_args()

    native = native_tokenizer()
    merges = fit_merges(native, (r['text'] for r in read_jsonl(args.train)), args.merges)
    student = build_tokenizer(native, merges)
    parts = decompose(merges)

    # Every new token must expand back to the native encoding and round-trip exactly.
    before = after = 0
    for row in read_jsonl(args.dev):
        native_ids = native.encode(row['text'], add_special_tokens=False).ids
        student_ids = student.encode(row['text'], add_special_tokens=False).ids
        assert [i for t in student_ids for i in parts.get(t, [t])] == native_ids
        assert student.decode(student_ids, skip_special_tokens=False) == row['text']
        before, after = before + len(native_ids), after + len(student_ids)

    write_json(args.out, {
        'first_new_id': FIRST_NEW_ID,
        'merges': [list(m) for m in merges],
        'tokens': [student.decode([FIRST_NEW_ID + k]) for k in range(len(merges))],
    })
    print(json.dumps({'merges': len(merges), 'dev_token_reduction': round(1 - after / before, 4)}))


if __name__ == '__main__':
    main()
