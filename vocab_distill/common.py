"""Shared constants and small file helpers."""
import gzip
import json
from pathlib import Path

MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
MODEL_REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
SEED = 17


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def native_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def chat_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def render_prompt(chat, prompt):
    return chat.apply_chat_template([{'role': 'user', 'content': prompt}],
                                    tokenize=False, add_generation_prompt=True)
