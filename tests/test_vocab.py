"""Checks that run without a GPU (they download Qwen's tokenizer once)."""
from pathlib import Path

from vocab_distill.common import native_tokenizer, read_jsonl
from vocab_distill.scoring import score
from vocab_distill.vocab import boundaries, build_tokenizer, decompose, load_vocab

ROOT = Path(__file__).resolve().parents[1]


def test_new_tokens_line_up_with_native_tokens():
    native = native_tokenizer()
    merges = load_vocab(ROOT / 'data/vocab256.json')
    student, parts = build_tokenizer(native, merges), decompose(merges)
    for text in ['भुगतान स्वीकृत नहीं हुआ।', 'Mixed text: यह फाइल 26 जून २०२६ को जमा नहीं हुई है।',
                 '<|im_start|>user\nनमस्ते<|im_end|>\n']:
        native_ids = native.encode(text, add_special_tokens=False).ids
        student_ids = student.encode(text, add_special_tokens=False).ids
        ends = boundaries(native_ids, student_ids, parts)
        assert ends[-1] == len(native_ids) - 1
        assert student.decode(student_ids, skip_special_tokens=False) == text
    hindi = 'भुगतान स्वीकृत नहीं हुआ। संपर्क पहचान है।'
    assert len(student.encode(hindi).ids) < len(native.encode(hindi).ids)


def test_shipped_predictions_give_the_paper_test_scores():
    rows = read_jsonl(ROOT / 'results/predictions/test_distill.jsonl.gz')
    content = {}
    for row in rows:
        content[row['task']] = content.get(row['task'], 0) + score(row)['content']
    assert content == {'belebele_hi': 491, 'belebele_en': 640, 'xquad_hi': 707, 'diverse_copy_hi': 92}
