"""Continued BPE: append new merges to Qwen's byte-level BPE tokenizer.

Every new token is a merge of existing tokens, so it always expands to a fixed
sequence of native tokens. Where a new token ends, a native token ends too.
That is why the unmodified model can act as an exact teacher (see `boundaries`).
"""
from collections import Counter, defaultdict
import heapq
import json

from tokenizers import Tokenizer

# Qwen3's embedding matrix has 151,936 rows (a few unused). New IDs start after all of them.
FIRST_NEW_ID = 151936


def pre_tokenize(tokenizer, text):
    """Split text the way the native tokenizer does. Merges never cross these pieces."""
    if tokenizer.normalizer is not None:
        text = tokenizer.normalizer.normalize_str(text)
    return [piece for piece, _ in tokenizer.pre_tokenizer.pre_tokenize_str(text)]


def replace_pair(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def fit_merges(native, texts, n_merges, min_frequency=2):
    """Greedy BPE over native token IDs. Returns [(left_id, right_id), ...] in rank order."""
    spec = json.loads(native.to_str())
    vocab = dict(spec['model']['vocab'])
    symbol = {i: s for s, i in vocab.items()}
    bpe = Tokenizer.from_str(native.to_str()).model
    special = [t['content'] for t in spec['added_tokens']]

    words = Counter()
    for text in texts:
        if any(s in text for s in special):
            raise ValueError('training text contains a special-token string')
        for piece in pre_tokenize(native, text):
            ids = tuple(t.id for t in bpe.tokenize(piece))
            if ids:
                words[ids] += 1

    # Pair counts plus an index of which words contain each pair, so a merge only
    # touches the words it changes instead of rescanning the corpus.
    sequences, weights = list(words), list(words.values())
    counts, where = Counter(), defaultdict(set)
    for i, seq in enumerate(sequences):
        for pair, n in Counter(zip(seq, seq[1:])).items():
            counts[pair] += weights[i] * n
            where[pair].add(i)
    heap = [(-n, pair) for pair, n in counts.items()]
    heapq.heapify(heap)

    merges = []
    while heap and len(merges) < n_merges:
        negative, pair = heapq.heappop(heap)
        if -negative != counts[pair]:
            continue  # stale heap entry
        if counts[pair] < min_frequency:
            break
        text = symbol[pair[0]] + symbol[pair[1]]
        if text in vocab:
            continue  # already a native token
        new_id = FIRST_NEW_ID + len(merges)
        symbol[new_id] = text
        vocab[text] = new_id
        merges.append(pair)

        changed = set()
        for i in list(where[pair]):
            before = sequences[i]
            after = replace_pair(before, pair, new_id)
            old_pairs, new_pairs = Counter(zip(before, before[1:])), Counter(zip(after, after[1:]))
            for p in old_pairs.keys() | new_pairs.keys():
                counts[p] += weights[i] * (new_pairs[p] - old_pairs[p])
                if new_pairs[p]:
                    where[p].add(i)
                else:
                    where[p].discard(i)
                changed.add(p)
            sequences[i] = after
        for p in changed:
            if counts[p] >= min_frequency:
                heapq.heappush(heap, (-counts[p], p))
    return merges


def build_tokenizer(native, merges):
    """The native tokenizer with `merges` appended. Native IDs and special tokens are unchanged."""
    spec = json.loads(native.to_str())
    vocab = spec['model']['vocab']
    symbol = {i: s for s, i in vocab.items()}
    # Pin special tokens to their IDs; otherwise loading renumbers them after the new rows.
    for token in spec['added_tokens']:
        vocab[token['content']] = token['id']
    for k, (left, right) in enumerate(merges):
        new_id = FIRST_NEW_ID + k
        symbol[new_id] = symbol[left] + symbol[right]
        vocab[symbol[new_id]] = new_id
        spec['model']['merges'].append([symbol[left], symbol[right]])
    tokenizer = Tokenizer.from_str(json.dumps(spec, ensure_ascii=False))
    for token in spec['added_tokens']:
        assert tokenizer.token_to_id(token['content']) == token['id']
    return tokenizer


def decompose(merges):
    """Map each new token ID to the native token IDs it is made of."""
    parts = {}
    for k, (left, right) in enumerate(merges):
        parts[FIRST_NEW_ID + k] = parts.get(left, [left]) + parts.get(right, [right])
    return parts


def boundaries(native_ids, student_ids, parts):
    """For each student token, the index of the native token where it ends."""
    ends, expanded = [], []
    for token in student_ids:
        expanded.extend(parts.get(token, [token]))
        ends.append(len(expanded) - 1)
    if expanded != list(native_ids):
        raise ValueError('student tokens do not expand to the native tokens')
    return ends


def load_vocab(path):
    """Read a vocabulary file written by scripts/fit_tokenizer.py."""
    with open(path) as f:
        data = json.load(f)
    assert data['first_new_id'] == FIRST_NEW_ID
    return [tuple(pair) for pair in data['merges']]
