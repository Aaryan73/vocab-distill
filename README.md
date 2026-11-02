# vocab-distill

Code and data for **Cheaper Hindi Without Forgetting: Boundary-Aligned Self-Distillation for Tokenizer Extension** ([paper](paper/main.pdf)).

Language models built around English split Hindi into many small tokens, so the same text costs several times more compute. Adding Hindi tokens to the vocabulary fixes the cost, but the model then has to learn what the new tokens mean, and it gets worse at Hindi along the way.

We add 256 Hindi tokens to Qwen3-4B-Instruct, cutting Hindi prompt length by 34%, and train the extended model to copy the original model's internal states. This recovers most of the lost accuracy.

## Results

Sealed TEST set, run once after the model was chosen on DEV. Scores count answers that contain the right content; strict-format scores are in the paper.

| Task (items) | Original model | Ours | Difference, points [95% CI] | Prompt tokens |
|---|---:|---:|---|---:|
| Belebele Hindi (724) | 505 | 491 | −1.9 [−4.5, +0.6] | −34% |
| Belebele English (716) | 640 | 640 | 0.0 [−0.6, +0.6] | 0% |
| XQuAD Hindi, answer found (998) | 757 | 707 | −5.0 [−7.5, −2.5] | −35% |
| Varied Hindi copying (150) | 121 | 92 | −19.3 [−27.3, −11.3] | −32% |

Reading comes back to near the original and English is untouched. Finding exact answer spans and character-exact copying are still worse; the paper says why.

## How it works

Every new token is a merge of tokens the model already has (continued BPE). So wherever a new token ends, an original token ends too, and at those points the original and extended models have read exactly the same text.

The original model, with the new rows and LoRA switched off, is therefore an exact teacher at every one of those points. Training combines three losses:

- cross-entropy on the original model's own answers, written with the new tokens
- matching hidden states at layers 9, 18, 27 and the last layer
- matching the next-token distribution through the original output head

Only the 256 new embedding rows and a rank-16 LoRA are trained (34M parameters). The run takes about 45 minutes on one H100.

## Layout

```
vocab_distill/      the method: vocabulary extension, model, losses, decoding, scoring
scripts/            one script per pipeline step (each has --help)
data/
  vocab256.json     the 256 merges used in the paper
  train/            teacher answers (training data) and the baselines' raw text
  eval/             DEV (822 items) and TEST (2,588 items) suites
results/
  predictions/      every DEV and TEST prediction from the paper
  dev.json, test.json   the paper's comparisons, from scripts/analyze.py
  runs.json         training stats and checkpoint hashes
  selection.json    the DEV-based model choice, recorded before TEST was run
paper/              LaTeX source and PDF
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

The model and tokenizer download from Hugging Face at a pinned revision.

## Check the paper's numbers (no GPU)

```bash
python scripts/analyze.py --control results/predictions/test_native.jsonl.gz \
    --model distill=results/predictions/test_distill.jsonl.gz

python scripts/analyze.py --control results/predictions/dev_native.jsonl.gz \
    --model distill=results/predictions/dev_distill.jsonl.gz \
    --model ce-only=results/predictions/dev_ce-only.jsonl.gz \
    --model distill-3m=results/predictions/dev_distill-3m.jsonl.gz \
    --model naive=results/predictions/dev_naive.jsonl.gz \
    --model native-lora=results/predictions/dev_native-lora.jsonl.gz

pytest tests
```

## Train and evaluate (one GPU, about 10 GB)

```bash
# the method, and the ablation without the alignment losses
python scripts/train.py --method distill --out work/distill
python scripts/train.py --method ce-only --out work/ce-only
python scripts/train.py --method distill --budget 3000000 --out work/distill-3m

# baselines
python scripts/train.py --method naive --out work/naive
python scripts/train.py --method native-lora --out work/native-lora

# predictions (no --checkpoint means the original model)
python scripts/evaluate.py --suite data/eval/dev.jsonl.gz --out work/dev_native.jsonl.gz
python scripts/evaluate.py --suite data/eval/dev.jsonl.gz --checkpoint work/distill/checkpoint.pt \
    --out work/dev_distill.jsonl.gz
python scripts/evaluate.py --suite data/eval/test.jsonl.gz --checkpoint work/distill/checkpoint.pt \
    --batch 16 --out work/test_distill.jsonl.gz

python scripts/analyze.py --control work/dev_native.jsonl.gz --model distill=work/dev_distill.jsonl.gz
```

Training sees the same sequences in the same order as the paper's runs, and the shipped data reproduces the paper's example and token counts exactly. Expect small differences in the final scores on other GPUs or library versions, since bf16 kernels are not bit-identical.

## Rebuilding the data from scratch

Everything under `data/` can be regenerated:

```bash
python scripts/prepare_data.py --exclude <benchmark passages .jsonl ...>   # Sangraha splits
python scripts/fit_tokenizer.py                                            # 256 merges
python scripts/make_prompts.py prompts                                     # training prompts
python scripts/make_prompts.py copy-probe                                  # DEV copy probe
python scripts/make_prompts.py copy-probe --docs work/data/test_hi.jsonl --seed 43 --split test --prefix tcopy
python scripts/teacher.py                                                  # teacher answers (GPU)
```

From the paper's splits, `fit_tokenizer.py` and `make_prompts.py` reproduce the shipped files byte for byte.

The teacher answers can differ slightly, because batched generation depends on padding. The paper's teacher ran with a batch size of 24 and then 20.

The Belebele and XQuAD items in the suites come from the pinned benchmark releases. They are shipped as-is, so DEV and TEST stay exactly as used.

## Data sources

- [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) (verified Hindi and English): training text and copy probes.
- [Belebele](https://huggingface.co/datasets/facebook/belebele) and [XQuAD](https://huggingface.co/datasets/google/xquad): evaluation.
- Model: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f`.

Each dataset keeps its own license.

## Citation

```bibtex
@misc{regmi2026cheaperhindi,
  title  = {Cheaper Hindi Without Forgetting: Boundary-Aligned Self-Distillation for Tokenizer Extension},
  author = {Regmi, Aaryan and Ande, Yajnesh and Castelino, Sheldon},
  year   = {2026},
  url    = {https://github.com/Aaryan73/vocab-distill}
}
```
