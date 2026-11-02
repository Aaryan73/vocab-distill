"""Download Hindi and English text from Sangraha and split it by document.

Each document goes to one split by a hash of its ID:
  tokenizer (25%)  fits the new merges
  adaptation (55%) training text and training prompts
  dev (10%), test (10%)  copy probes

Documents that overlap benchmark text (8-word shingles) are dropped. Pass the benchmark
passages as JSONL files with a `text` field via --exclude.

Note: the repo already ships everything later steps need (data/), so this script is
only for rebuilding the pipeline from scratch.
"""
import argparse
from collections import Counter
import hashlib

from vocab_distill.common import read_jsonl, write_jsonl

SANGRAHA_REVISION = '8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split_of(document_id):
    bucket = int(digest(document_id)[:8], 16) % 100
    return 'tokenizer' if bucket < 25 else 'adaptation' if bucket < 80 else 'dev' if bucket < 90 else 'test'


def shingles(text, size=8):
    words = ' '.join(text.casefold().split()).split()
    if len(words) < size:
        return {digest(' '.join(words))} if words else set()
    return {digest(' '.join(words[i:i + size]))[:24] for i in range(len(words) - size + 1)}


class Overlap:
    """Flags texts whose 8-word shingles are at least half seen before."""

    def __init__(self):
        self.exact, self.pieces = set(), set()

    def add(self, text):
        self.exact.add(digest(' '.join(text.casefold().split())))
        self.pieces |= shingles(text)

    def seen(self, text):
        parts = shingles(text)
        return (digest(' '.join(text.casefold().split())) in self.exact
                or bool(parts) and len(parts & self.pieces) / len(parts) >= 0.5)


def mix(hindi, english, hindi_fraction=0.8):
    """Interleave documents so about 80% of bytes are Hindi."""
    streams, totals = {'hi': iter(hindi), 'en': iter(english)}, Counter()
    while True:
        language = 'hi' if totals['hi'] <= hindi_fraction * sum(totals.values()) else 'en'
        row = next(streams[language], None)
        if row is None:
            return
        totals[language] += len(row['text'].encode())
        yield row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='work/data')
    p.add_argument('--bytes-per-language', type=int, default=20_000_000)
    p.add_argument('--exclude', nargs='*', default=[], help='JSONL files of benchmark text to keep out')
    args = p.parse_args()
    from datasets import load_dataset

    benchmark, seen = Overlap(), Overlap()
    for path in args.exclude:
        for row in read_jsonl(path):
            benchmark.add(row['text'])

    rows = {(s, l): [] for s in ['tokenizer', 'adaptation', 'dev', 'test'] for l in ['hi', 'en']}
    for language, folder in [('hi', 'hin'), ('en', 'eng')]:
        url = (f'https://huggingface.co/datasets/ai4bharat/sangraha/resolve/{SANGRAHA_REVISION}'
               f'/verified/{folder}/data-0.parquet')
        total = 0
        for item in load_dataset('parquet', data_files=url, split='train', streaming=True):
            text = item['text']
            # Chat-template strings must never appear in ordinary training text.
            if not text.strip() or len(text) > 30000 or '<|' in text or '<think>' in text or '<tool_call>' in text:
                continue
            if benchmark.seen(text) or seen.seen(text):
                continue
            split = split_of(str(item['doc_id']))
            rows[split, language].append({'id': str(item['doc_id']), 'split': split, 'language': language, 'text': text})
            seen.add(text)
            total += len(text.encode())
            if total >= args.bytes_per_language:
                break

    for (split, language), records in rows.items():
        write_jsonl(f'{args.out}/{split}_{language}.jsonl', records)
        print(f'{split}_{language}: {len(records)} documents')
    write_jsonl(f'{args.out}/adaptation_mixed.jsonl', mix(rows['adaptation', 'hi'], rows['adaptation', 'en']))


if __name__ == '__main__':
    main()
