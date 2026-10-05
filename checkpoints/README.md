# Checkpoints

Both checkpoints load with `laya.agent.Agent("<folder>")`. The weights (`model.safetensors`, 842 MB each) are stored with **Git LFS**. If they come out as small text pointer files after a clone, run `git lfs pull`.

| folder | what it is |
|---|---|
| `laya-vanilla/` | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) at revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, English checkpoint, unmodified weights. `MODEL_CARD.md` is the vendor's model card. |
| `laya-finetuned/` | `laya-vanilla` fine-tuned on I-GUIDE decisions with `training/finetune.py`, on `data/train/router_train.jsonl` + `data/train/tool_train_rows.jsonl`. 4 epochs on one RTX 3060, about 76 min. Training metadata is in `iguide_finetune.json`; `iguide_questions.json` holds the exact router question wording it was trained on. |

## License

Laya is released under the **Apache License 2.0** (see `laya-vanilla/MODEL_CARD.md`); the full text is in `LICENSE-APACHE-2.0.txt`.

- **`laya-finetuned/` is a modified version (a derivative work).** Its weights were changed by fine-tuning, and its temperatures were refitted on held-out training data. It is distributed under the same license.
- **`laya-vanilla/` is redistributed unchanged.** Its files are byte-identical to the Hugging Face revision above.
