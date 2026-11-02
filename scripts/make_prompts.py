"""Build the distillation prompts and the Hindi copy probes.

prompts: 3,000 instruction prompts (2,400 Hindi, 600 English) over six tasks, plus
1,000 raw text segments, all from the adaptation split. No benchmark text is used,
and every template is worded differently from the evaluation prompts.

copy-probe: one sentence from each of 150 held-out documents, under three
"copy this exactly" instructions. The paper uses seed 29 on dev_hi (DEV) and
seed 43 on test_hi (TEST).
"""
import argparse
from collections import Counter
import random
import re

from vocab_distill.common import read_jsonl, write_jsonl

SENTENCE_END = re.compile(r'(?<=[।.!?])\s+')

TASKS = {
    'hi': {
        'summary': 'नीचे दिए गए लेख को पढ़कर उसका सार दो-तीन वाक्यों में हिंदी में बताइए।\n\n{p}',
        'facts': '{p}\n\nइस पाठ से तीन महत्वपूर्ण तथ्य बिंदुओं में लिखिए।',
        'extract': 'पाठ:\n{p}\n\nइस पाठ का {k} वाक्य बिना किसी बदलाव के लिखिए। और कुछ न जोड़ें।',
        'translate': 'इस हिंदी पाठ के पहले दो वाक्यों का अंग्रेज़ी में अनुवाद कीजिए:\n\n{p}',
        'topic': '{p}\n\nयह पाठ मुख्य रूप से किस बारे में है? एक वाक्य में उत्तर दीजिए।',
        'cloze': '{p}\n\nरिक्त स्थान (____) में कौन सा शब्द आएगा?\n{o}\nउत्तर में सिर्फ विकल्प का अक्षर दीजिए।',
    },
    'en': {
        'summary': 'Read the article below and summarize it in two or three sentences.\n\n{p}',
        'facts': '{p}\n\nList three important facts from this text as bullet points.',
        'extract': 'Text:\n{p}\n\nReproduce the {k} sentence of this text without any change. Add nothing else.',
        'translate': 'Translate the first two sentences of this text into Hindi:\n\n{p}',
        'topic': '{p}\n\nWhat is this text mainly about? Answer in one sentence.',
        'cloze': '{p}\n\nWhich word fills the blank (____)?\n{o}\nAnswer with the option letter only.',
    },
}
ORDINALS = {'hi': ['पहला', 'दूसरा', 'तीसरा'], 'en': ['first', 'second', 'third']}
MAX_NEW_TOKENS = {'summary': 160, 'facts': 160, 'extract': 128, 'translate': 160, 'topic': 64, 'cloze': 8}
COPY_TEMPLATES = [
    'नीचे दिया गया वाक्य बिना कोई अक्षर बदले दोबारा लिखिए:\n{s}',
    '{s}\n\nऊपर वाले वाक्य की हूबहू नकल कीजिए। कुछ और मत लिखिए।',
    'Copy this Hindi sentence exactly as written:\n{s}',
]


def passage(text, rng, low=300, high=900):
    """A run of 3+ consecutive sentences, 300-900 characters long, or None."""
    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip() and '<|' not in s]
    if len(sentences) < 4:
        return None
    start = rng.randrange(0, max(1, len(sentences) - 3))
    chosen = []
    for s in sentences[start:]:
        if sum(map(len, chosen)) + len(s) > high:
            break
        chosen.append(s)
    return chosen if len(chosen) >= 3 and len(' '.join(chosen)) >= low else None


def make_prompts(args):
    rng = random.Random(args.seed)
    docs = {'hi': read_jsonl(args.hindi), 'en': read_jsonl(args.english)}
    # Cloze distractors come from words in the first 400 documents.
    words = {lang: [w for d in docs[lang][:400] for w in d['text'].split() if 3 <= len(w) <= 12 and w.isalpha()]
             for lang in docs}

    rows = []
    for language, count in [('hi', args.hindi_chat), ('en', args.english_chat)]:
        pool = docs[language][:]
        rng.shuffle(pool)
        names = list(TASKS[language])
        for doc in pool:
            if sum(r['language'] == language for r in rows) >= count:
                break
            sentences = passage(doc['text'], rng)
            if not sentences:
                continue
            task = names[len(rows) % len(names)]
            fill = {'p': ' '.join(sentences)}
            if task == 'extract':
                fill['k'] = ORDINALS[language][rng.randrange(3)]
            if task == 'cloze':
                sentence = sentences[1]
                candidates = [w for w in sentence.split() if 3 <= len(w) <= 12 and w.isalpha()]
                if not candidates:
                    continue
                answer = rng.choice(candidates)
                options = [answer] + rng.sample([w for w in words[language] if w != answer], 3)
                rng.shuffle(options)
                sentences[1] = sentence.replace(answer, '____', 1)
                fill['p'] = ' '.join(sentences)
                fill['o'] = '\n'.join(f'{letter}. {w}' for letter, w in zip('ABCD', options))
            rows.append({'id': f'{language}-{task}-{doc["id"][:16]}', 'source_id': doc['id'],
                         'language': language, 'task': task, 'kind': 'chat',
                         'max_new_tokens': MAX_NEW_TOKENS[task], 'prompt': TASKS[language][task].format(**fill)})

    used = {r['source_id'] for r in rows}
    for language, share in [('hi', 0.8), ('en', 0.2)]:
        pool = [d for d in docs[language] if d['id'] not in used]
        rng.shuffle(pool)
        for doc in pool[:int(args.raw * share)]:
            rows.append({'id': f'{language}-raw-{doc["id"][:16]}', 'source_id': doc['id'],
                         'language': language, 'task': 'raw', 'kind': 'raw', 'text': doc['text'][:1500]})
    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    print(Counter(f"{r['language']}/{r['task']}" for r in rows))


def make_copy_probe(args):
    rng = random.Random(args.seed)
    docs = read_jsonl(args.docs)
    rng.shuffle(docs)
    seen, items = set(), []
    for doc in docs:
        candidates = [s.strip() for s in SENTENCE_END.split(doc['text'])
                      if 40 <= len(s.strip()) <= 200 and '\n' not in s.strip() and '<|' not in s]
        if not candidates:
            continue
        sentence = rng.choice(candidates)
        if sentence in seen:
            continue
        seen.add(sentence)
        i = len(items)
        items.append({'id': f'{args.prefix}-{i}', 'task': 'diverse_copy_hi', 'cluster': doc['id'],
                      'split': args.split, 'prompt': COPY_TEMPLATES[i % 3].format(s=sentence),
                      'reference': sentence, 'max_new_tokens': 256, 'template': i % 3})
        if len(items) >= args.items:
            break
    write_jsonl(args.out, items)
    print(len(items), 'copy items ->', args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prompts')
    a.add_argument('--hindi', default='work/data/adaptation_hi.jsonl')
    a.add_argument('--english', default='work/data/adaptation_en.jsonl')
    a.add_argument('--out', default='work/prompts.jsonl')
    a.add_argument('--seed', type=int, default=17)
    a.add_argument('--hindi-chat', type=int, default=2400)
    a.add_argument('--english-chat', type=int, default=600)
    a.add_argument('--raw', type=int, default=1000)
    a = sub.add_parser('copy-probe')
    a.add_argument('--docs', default='work/data/dev_hi.jsonl')
    a.add_argument('--out', default='work/copy_dev.jsonl')
    a.add_argument('--split', default='dev')
    a.add_argument('--prefix', default='dcopy', help='item ID prefix (the paper used tcopy for TEST)')
    a.add_argument('--items', type=int, default=150)
    a.add_argument('--seed', type=int, default=29)
    args = p.parse_args()
    make_prompts(args) if args.command == 'prompts' else make_copy_probe(args)


if __name__ == '__main__':
    main()
