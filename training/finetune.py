"""Fine-tune Laya on the search router's two decisions (graph? spatial?), one GPU or CPU.

Follows the vendor's recipe (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb in
github.com/NandhaKishorM/laya) minus DDP: soft cross-entropy plus a proper-scoring-rule policy
gradient term, then per-type temperatures fitted on a slice held out of training. The output
directory has the layout `laya.Agent` loads, so the sidecar serves it with LAYA_MODEL_DIR=<out>.

Training sequences are built with the SAME functions the Agent uses at inference
(`Agent._to_internal`, `build_sequence`, the state tokenized once), so a question is seen in
training exactly as it will be asked in serving. The question wording is part of the model:
it is written to <out>/iguide_questions.json, and a client must send those questions verbatim.

    python training/finetune.py --base laya_models/laya --train data/train/router_train.jsonl data/train/tool_train_rows.jsonl --out laya_models/my-finetune

`--train` takes one or more JSONL files, in either of two row formats:

  {"query": ..., "graph": bool, "spatial": bool}     the search-router questions (QUESTIONS);
                                                      a float in [0, 1] is a soft P(true)
  {"state": ..., "questions": {qid: qdef}, "targets": {qid: [p per option]},
   "max_len": int?, "head_max_len": int?}            any typed question, e.g. the tool-choice
                                                      rows make_tool_training_data.py writes

Per-row max_len/head_max_len matter for high-cardinality choice: 53 options do not fit the
default 192-token option budget. Batches are packed by token count, not row count.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import Agent, _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

# The two questions the fine-tuned model answers. Plain, criteria-free wording (the style that
# ranked best zero-shot among scripts/eval_laya_router.py VARIANTS), widened to say what the
# labels mean: graph also covers popularity and links between resources, spatial also covers
# distance and map areas (RUBRIC in make_training_data.py).
QUESTIONS = {
    "graph": {"type": "noul",
              "instructions": "Does this query name a specific person, organization or tag, ask "
                              "what is popular, or ask for resources linked to another resource?"},
    "spatial": {"type": "noul",
                "instructions": "Does this query mention a specific place, coordinates, or a "
                                "distance or area on a map?"},
}


def encode(tok, cfg, query, questions, max_len=None, head_max_len=None):
    """Sequences for one query, built as Agent._encode_state builds them at inference."""
    max_len = max_len or cfg.get("max_len", 512)
    head_max_len = head_max_len or cfg.get("head_max_len", 192)
    state_ids = tok(query.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
    out = {}
    for qid, qdef in questions.items():
        q = Agent._to_internal(qdef)
        seq, markers = build_sequence(tok, query, q, max_len, head_max_len, state_ids=state_ids)
        assert len(markers) == len(render_options(q)), (qid, query)
        out[qid] = {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}
    return out


def load_items(paths, tok, cfg, questions):
    """Items tagged with their row, so the calibration split can hold out whole rows."""
    items, row_no = [], 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "questions" in row:
                    enc = encode(tok, cfg, row["state"], row["questions"],
                                 row.get("max_len"), row.get("head_max_len"))
                    for qid, it in enc.items():
                        t = [float(x) for x in row["targets"][qid]]
                        s = sum(t)
                        t = [x / s for x in t]
                        items.append({**it, "target": t, "label": t.index(max(t)), "row": row_no})
                else:
                    for qid, it in encode(tok, cfg, row["query"], questions).items():
                        p = float(row[qid])
                        items.append({**it, "target": [1.0 - p, p], "label": int(p >= 0.5), "row": row_no})
                row_no += 1
    return items, row_no


def token_batches(items, max_tokens, max_rows, rnd):
    """Shuffle, then pack greedily so a batch's PADDED size stays under max_tokens."""
    order = items[:]
    rnd.shuffle(order)
    batches, cur, longest = [], [], 0
    for it in order:
        L = max(longest, len(it["ids"]))
        if cur and (L * (len(cur) + 1) > max_tokens or len(cur) >= max_rows):
            batches.append(cur)
            cur, L = [], len(it["ids"])
        cur.append(it)
        longest = L
    if cur:
        batches.append(cur)
    return batches


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_temperature(pairs):
    """Scalar T minimising soft NLL of softmax(z / T) on held-out (logits, target) pairs."""
    kmax = max(len(z) for z, _ in pairs)
    Z = torch.full((len(pairs), kmax), -1e4)
    T = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    # Laya's loader clamps temperatures to [0.5, 5] and warns outside it; fit inside that range.
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="Laya checkpoint dir (rl_agent_config.json, model.safetensors, ...)")
    ap.add_argument("--train", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=6144, help="padded tokens per micro-batch")
    ap.add_argument("--micro-batch", type=int, default=16, help="max rows per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    ap.add_argument("--calib-frac", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N micro-batches (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    _fix_tokenizer_config(args.base)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.base, "tokenizer"))
    with open(os.path.join(args.base, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    items, n_rows = load_items(args.train, tok, cfg, QUESTIONS)
    # Hold out the calibration slice BY ROW, so no row's other question leaks into training.
    order = list(range(n_rows))
    random.Random(args.seed).shuffle(order)
    n_cal = max(20, int(n_rows * args.calib_frac))
    cal_rows = set(order[:n_cal])
    calib = [it for it in items if it["row"] in cal_rows]
    train = [it for it in items if it["row"] not in cal_rows]
    by_type = {t: sum(it["qtype"] == v for it in train) for t, v in QTYPES.items()}
    print(f"{n_rows} rows -> {len(train)} train / {len(calib)} calibration sequences; by type {by_type}; "
          f"longest {max(len(it['ids']) for it in items)} tokens")

    model = build_model(cfg, encoder_dir=os.path.join(args.base, "encoder"))
    model.load_state_dict(load_file(os.path.join(args.base, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_encoder},
                             {"params": head, "lr": args.lr_head}], weight_decay=0.01)
    n_batches = len(token_batches(train, args.max_tokens, args.micro_batch, random.Random(0)))
    updates = max(1, math.ceil(n_batches / args.grad_accum) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)
    G, SIGMA0, SIGMA1 = 4, 0.4, 0.1   # recipe: group size and exploration-noise schedule

    def forward(batch):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_cuda):
            return model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                         batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                         batch["qtype"].to(device))

    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        batches = token_batches(train, args.max_tokens, args.micro_batch, random.Random(args.seed + epoch))
        sigma = SIGMA0 + (SIGMA1 - SIGMA0) * (epoch / max(1, args.epochs - 1))
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, chunk in enumerate(batches):
            batch = collate(chunk, tok.pad_token_id)
            logits, act = forward(batch)
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((G,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            nb += 1
            if nb % args.grad_accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot += loss_ce.item()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"  epoch {epoch + 1} step {step} ce {loss_ce.item():.4f} "
                      f"reward {r.mean().item():.3f} {time.time() - t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        print(f"=== epoch {epoch + 1}/{args.epochs} mean ce {tot / max(1, nb):.4f} ({time.time() - t0:.0f}s)")
        if args.max_steps and step >= args.max_steps:
            break

    # Temperatures on the held-out slice, one per question type present there; a type with no
    # held-out items keeps the base checkpoint's value rather than a placeholder.
    model.eval()
    pairs = {}
    with torch.no_grad():
        for chunk in token_batches(calib, args.max_tokens, 32, random.Random(1)):
            lg, _ = forward(collate(chunk, tok.pad_token_id))
            lg = lg.float().cpu().numpy()
            for i, it in enumerate(chunk):
                pairs.setdefault(it["qtype"], []).append((lg[i, :len(it["markers"])], it["target"]))
    temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
    for qt, ps in pairs.items():
        temps[qt] = fit_temperature(ps)
    names = {v: k for k, v in QTYPES.items()}
    print("fitted temperatures:", {names[qt]: round(temps[qt], 3) for qt in pairs})

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(args.out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.out, "encoder"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))
    cfg.update({"fine_tuned": True, "model_name": "laya-iguide-router", "temperature": temps})
    # Agent prefers a matching temperature_by_options bucket over `temperature`; drop the base
    # buckets so the values fitted here are the ones that apply.
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "iguide_questions.json"), "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, indent=2)
    shas = {}
    for path in args.train:
        with open(path, "rb") as f:
            shas[os.path.abspath(path)] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(args.out, "iguide_finetune.json"), "w") as f:
        json.dump({"base": os.path.abspath(args.base), "train_sha256": shas, "rows": n_rows,
                   "calibration_rows": n_cal, "epochs": args.epochs, "steps": step, "device": str(device),
                   "amp": str(amp_dtype) if use_cuda else "fp32", "seconds": round(time.time() - t0)},
                  f, indent=2)
    print(f"saved {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
