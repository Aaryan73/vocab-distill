"""Strict and content scores, plus paired statistics.

Strict scores check the exact output format; content scores check whether the right
answer is there at all. The paper reports both because strict scores reward short,
well-formatted answers even when the model understands less.
"""
from collections import Counter
import math
import re
import string
import unicodedata

import numpy as np

COPY_TASKS = {'exact_details', 'controlled_copy', 'diverse_copy_hi'}


def normalize_answer(text):
    """SQuAD normalization, extended to Unicode punctuation."""
    text = text.casefold()
    text = ''.join(c for c in text if not (c in string.punctuation or unicodedata.category(c).startswith('P')))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def overlap_f1(prediction, reference):
    if not prediction or not reference:
        return float(prediction == reference)
    common = sum((Counter(prediction) & Counter(reference)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(prediction), common / len(reference)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction, references):
    return max(overlap_f1(normalize_answer(prediction).split(), normalize_answer(r).split()) for r in references)


def character_f1(prediction, reference):
    return overlap_f1(list(normalize_answer(prediction).replace(' ', '')),
                      list(normalize_answer(reference).replace(' ', '')))


def score(row):
    """Scores for one prediction row. `strict` and `content` are always present (0 or 1)."""
    task, prediction = row['task'], row['prediction']
    if task in COPY_TASKS:
        text, reference = prediction.strip(), row['reference']
        return {'strict': float(text == reference), 'content': float(reference in text),
                'char_f1': character_f1(text, reference)}
    if task == 'xquad_hi':
        references = row.get('references', [row['reference']])
        normalized = normalize_answer(prediction)
        return {'strict': float(any(normalized == normalize_answer(r) for r in references)),
                'content': float(any(r in prediction for r in references)),
                'token_f1': token_f1(prediction, references)}
    if task.startswith('belebele'):
        letter = re.match(r'\s*\(?([ABCD])\b', prediction)
        return {'strict': float(prediction.strip() == row['reference']),
                'content': float(bool(letter) and letter.group(1) == row['reference'])}
    raise ValueError(f'unknown task {task}')


def clustered_bootstrap(differences, clusters, resamples=10000, seed=17):
    """Mean paired difference with a 95% CI that resamples passages, not questions."""
    groups = {}
    for d, c in zip(differences, clusters):
        groups.setdefault(c, []).append(d)
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = [sums[ix].sum() / counts[ix].sum()
             for ix in (rng.integers(len(sums), size=len(sums)) for _ in range(resamples))]
    lower, upper = np.quantile(draws, [0.025, 0.975])
    mean = sums.sum() / counts.sum()
    if len(set(differences)) == 1:
        # Every pair agrees, so the bootstrap gives a zero-width interval. Widen it by
        # the largest disagreement rate that would still go unseen 97.5% of the time.
        effective = max(1, int(counts.sum() ** 2 / (counts ** 2).sum()))
        unseen = 1 - 0.025 ** (1 / effective)
        lower, upper = min(lower, mean - (1 + mean) * unseen), max(upper, mean + (1 - mean) * unseen)
    return float(mean), float(lower), float(upper)


def mcnemar(only_a, only_b):
    """Exact two-sided McNemar test from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
