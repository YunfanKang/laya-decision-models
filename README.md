# Laya decision models for I-GUIDE AI

The I-GUIDE agent makes many small decisions per request: does a search need the knowledge graph, does it need the spatial search, which analysis tool runs first? Today an LLM makes each one by writing JSON that code parses. This repo shows a **decision model** doing the same job. A decision model answers questions with a fixed set of answers directly, as probabilities, in one forward pass. The model is [Laya](https://huggingface.co/convaiinnovations/laya) (421M parameters, Apache-2.0), used as downloaded and after fine-tuning on I-GUIDE decisions.

**Start with [`laya_decision_models.ipynb`](laya_decision_models.ipynb).** It covers:

- how an LLM makes these decisions today;
- how a decision model works and how it differs;
- example queries answered by vanilla and fine-tuned Laya, side by side.

The outputs are saved, so it reads fine on GitHub without running anything.

## Results in one table

All numbers are on hand-labelled test sets that weren't used in training:

| | Vanilla Laya | Fine-tuned Laya | LLM baseline |
|---|---|---|---|
| Search routing: graph / spatial accuracy (87 queries) | 0.63 / 0.67 | **0.95 / 0.99** | 0.87 / 0.93 (gpt-5.6-luna) |
| First tool, exact match (42 held-out requests) | 0.10 | **0.79** | 0.67 (gpt-4o) |
| Time per routing decision | | ~0.1 s GPU · ~0.5–1 s CPU | ~1.6 s (API) |

The full evaluation, with its caveats, is in [`docs/evaluation.md`](docs/evaluation.md). The main caveats:

- **The fine-tuned model is overconfident.** On the yes/no questions it answers close to 0 or 1 even when it's wrong.
- **Typos are its weakest spot.**
- **The tool test favours Laya.** gpt-4o reaches 0.95 if "inspect the file first" also counts as right.

## Contents

```
laya_decision_models.ipynb   the notebook
requirements.txt
data/
  questions.json             the questions exactly as each model is asked them (router wording, 53 tool options)
  eval/                      hand-labelled test sets: 96 routing queries, 89 + 42 tool-selection requests
  train/                     the exact training rows of the fine-tuned checkpoint
    router_train.jsonl         3,541 catalog queries labelled graph/spatial
    tool_train_rows.jsonl      2,861 requests labelled with the acceptable first tools
training/finetune.py         the fine-tuning script (single GPU)
docs/evaluation.md           the full evaluation write-up
```

The fine-tuned weights (846 MB) are too large for git. They are attached to this repo's **GitHub release** as `laya_ft3_checkpoint.zip`.

## Run it

Python 3.10 or newer. A CPU is enough; you need about 4 GB of RAM.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.14"   # CPU-only machines
pip install -r requirements.txt
```

Get the fine-tuned checkpoint, either way:

- **Download and unzip it here.** Download `laya_ft3_checkpoint.zip` from the release and unzip it in the repo folder, which creates `laya_models/laya-ft-tools-v2/`.
- **Let the notebook fetch it.** Set `LAYA_FT3_URL` to the zip's download link.

Vanilla Laya downloads from Hugging Face on first run (843 MB, pinned to revision `55cf4c4`). Then open the notebook and run all cells. On a 2-CPU machine the whole notebook takes about three minutes, plus the first download.

## Reproduce the fine-tuning

This needs a GPU with 12 GB or more. About 75 minutes on an RTX 3060:

```bash
python training/finetune.py --base laya_models/laya \
    --train data/train/router_train.jsonl data/train/tool_train_rows.jsonl \
    --out laya_models/my-finetune
```

The recipe is the Laya vendor's own, ported to a single GPU:
- **Loss:** soft cross-entropy plus a policy-gradient term with a proper-scoring reward.
- **Calibration:** a temperature per question type, fitted on a held-out slice of the training data.
- **Length:** 4 epochs. The training loss reaches zero early, so fewer epochs is the obvious next experiment for better-calibrated confidence.

## Where the data came from

- **Training requests:** written by an LLM (gpt-5.6-luna).
- **Training labels:** assigned by a second LLM (gpt-5.2) against a written rubric. That rubric is the graph/spatial definitions in `docs/evaluation.md`.
- **Test labels:** written by hand by one person, following the same rubric.
- **Keeping the tests unseen:** any generated request close to a test query was dropped before training.

The original work, including the agent pipeline these decisions come from, is in the I-GUIDE `iguide-ai` repository, branch `laya-sidecar-evaluation`.
