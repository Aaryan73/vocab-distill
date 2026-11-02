"""Compare each model's predictions with the unmodified model's, item by item.

For every task and both scores (strict, content) this reports correct counts, the paired
difference in points with a 95% passage-clustered bootstrap CI, items lost and gained,
and an exact McNemar p-value. It also reports how many fewer prompt tokens the model used.

Example (reproduces the paper's TEST table from the shipped predictions):
  python scripts/analyze.py --control results/predictions/test_native.jsonl.gz \
      --model distill=results/predictions/test_distill.jsonl.gz --out work/test.json
"""
import argparse

from vocab_distill.common import read_jsonl, write_json
from vocab_distill.scoring import clustered_bootstrap, mcnemar, score


def compare(control, model, ids):
    report = {'items': len(ids),
              'prompt_token_reduction': 1 - sum(model[i]['input_tokens'] for i in ids)
                                        / sum(control[i]['input_tokens'] for i in ids)}
    for kind in ('strict', 'content'):
        a = [model[i]['score'][kind] for i in ids]
        b = [control[i]['score'][kind] for i in ids]
        diff, low, high = clustered_bootstrap([x - y for x, y in zip(a, b)], [control[i]['cluster'] for i in ids])
        lost = sum(y > x for x, y in zip(a, b))
        gained = sum(x > y for x, y in zip(a, b))
        report[kind] = {'correct': int(sum(a)), 'control_correct': int(sum(b)),
                        'difference_points': round(100 * diff, 2), 'ci95': [round(100 * low, 2), round(100 * high, 2)],
                        'lost': lost, 'gained': gained, 'mcnemar_p': mcnemar(lost, gained)}
    return report


def load(path):
    return {r['id']: {**r, 'score': score(r)} for r in read_jsonl(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--control', required=True, help='predictions of the unmodified model')
    p.add_argument('--model', action='append', required=True, help='name=predictions file; repeatable')
    p.add_argument('--out')
    args = p.parse_args()

    control = load(args.control)
    tasks = sorted({r['task'] for r in control.values()})
    report = {}
    for spec in args.model:
        name, path = spec.split('=', 1)
        model = load(path)
        report[name] = {t: compare(control, model, [i for i, r in control.items() if r['task'] == t and i in model])
                        for t in tasks}
        print(f'\n{name} vs native')
        print(f'{"task":18} {"score":8} {"model":>6} {"native":>6} {"diff":>7}  {"95% CI":>18}  {"p":>6}  prompt tokens')
        for task, r in report[name].items():
            for kind in ('strict', 'content'):
                s = r[kind]
                ci = f'[{s["ci95"][0]:+.2f}, {s["ci95"][1]:+.2f}]'
                print(f'{task:18} {kind:8} {s["correct"]:6d} {s["control_correct"]:6d} {s["difference_points"]:+7.2f}'
                      f'  {ci:>18}  {s["mcnemar_p"]:6.3f}  {-100 * r["prompt_token_reduction"]:+.1f}%')
    if args.out:
        write_json(args.out, report)


if __name__ == '__main__':
    main()
