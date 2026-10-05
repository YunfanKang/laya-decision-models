> Full evaluation write-up, copied from the iguide-ai repo (`docs/laya-evaluation.md`, branch `laya-sidecar-evaluation`). File paths in it refer to that repo; `outputs/` there is not part of this repo.

# Laya evaluation — progress and findings

Can [Laya](https://huggingface.co/convaiinnovations/laya) (a 421M non-generative decision model,
ModernBERT-large + a typed decision head, Apache-2.0) replace LLM decisions in this agent?
Work log from 2026-09-25 to 2026-09-29. Every number below comes from a file under `outputs/`
(git-ignored) produced by a script in this repo; the commands are at the end.

## Status in one paragraph

Zero-shot, Laya is not usable here. Fine-tuned on synthetic data it matches or beats the LLM on
the two decisions we built test sets for — but **both of those test sets model the decision
wrongly**: the graph/spatial router is not on the map-UI path at all, and the tool-selection input
gave Laya file metadata the real model never sees. Before more training, the target has to be
re-chosen from the decisions the pipeline actually makes (§9); the strongest candidate is the
supervisor's `decide()`.

## 1. What exists

| Path | What it is |
|---|---|
| `laya-server/Dockerfile`, `serve.py` | Sidecar image: CPU torch, `laya==0.3.20`, English checkpoint baked at revision `55cf4c4`, `HF_HUB_OFFLINE=1`, every request routed to the English checkpoint. **Never built** (no Docker on the dev machine); `serve.py` itself was run locally. |
| `docker-compose.yml` → `laya` | Opt-in (`--profile laya`), host port `127.0.0.1:8010`, unauthenticated by default. Nothing calls it. |
| `laya-server/finetune.py` | Single-GPU port of the vendor's recipe (soft CE + proper-scoring policy gradient, held-out temperature fit). Mixed row formats, per-row lengths, token-budget batching. Output loads in `laya.Agent`; train/serve encoding parity verified. |
| `laya-server/make_training_data.py` | Graph/spatial queries (gpt-5.6-luna) labelled against a written rubric (gpt-5.2). |
| `laya-server/make_tool_training_data.py` | Tool-selection requests + labels (v1 and v2 labelling prompt). |
| `laya-server/clean_training_data.py` | Drops/repairs self-contradictory rows; logs every change. |
| `scripts/eval_laya_router.py` | Graph/spatial: LLM router vs regex fallback vs Laya (sidecar HTTP). |
| `scripts/eval_laya_tools.py` | Tool selection: LLM with real tool schemas vs Laya (full 53-way or MiniLM shortlist). |
| `scripts/pipeline_label.py` | Labels requests with the REAL pipeline: uploads stand-in files, sends the map UI's exact payload to a local `/agent/chat/stream`, parses the SSE trace. |
| `scripts/make_router_shifts.py`, `router_shift_report.py` | Five label-preserving shifts of the router gold set and the shift report. |
| `scripts/router_eval_queries.jsonl` | 96 graph/spatial gold queries (9 marked ambiguous). |
| `scripts/tool_eval_queries.jsonl`, `tool_eval_holdout.jsonl` | 89 + 42 tool-selection gold requests, each with every acceptable first tool. |

All gold labels were written by one person (the agent doing this work) following written criteria;
treat them as a consistent reading, not ground truth.

## 2. Graph/spatial routing (the `_llm_route` decision)

87 unambiguous queries. LLM = gpt-5.6-luna through `LLMDecisionEngine`. Fine-tunes use a fixed
0.5 cut-off; nothing about them was fitted on gold.

| Router | graph | spatial | both flags right | p50 latency |
|---|---|---|---|---|
| LLM router | 0.874 | 0.931 | 0.816 | 1,565 ms (API) |
| Regex fallback (`detect_pattern`) | 0.690 | 0.586 | — | ~0 |
| Laya zero-shot, 0.5 cut-off | 0.621 | 0.678 | 0.425 | ~330 ms CPU |
| Laya zero-shot, cut-off tuned on gold (optimistic) | 0.805 | 0.908 | 0.724 | |
| Fine-tune 1 (graph/spatial data only) | 0.897 | 0.989 | 0.885 | ~307 ms CPU |
| Fine-tune 2 (tool + graph/spatial data) | **0.943** | **0.989** | **0.931** | |

Robustness (graph accuracy, fixed cut-off, shifts never seen in training):

| | clean | lower | typo | wrapped | preamble | paraphrase |
|---|---|---|---|---|---|---|
| LLM router | 0.874 | 0.874 | 0.897 | 0.897 | 0.897 | 0.897 |
| Laya zero-shot (tuned cut-off) | 0.805 | 0.747 | 0.667 | 0.782 | 0.690 | 0.713 |
| Fine-tune 1 | 0.897 | 0.874 | 0.724 | 0.885 | 0.874 | 0.920 |
| Fine-tune 2 | 0.943 | 0.943 | **0.828** | 0.954 | 0.897 | 0.954 |

Findings:
- Zero-shot Laya ranks reasonably (spatial AUROC 0.94) but is uncalibrated: the useful cut-offs
  were 0.026/0.113, and every rewording moved scores across them (up to 34% of decisions flipped).
  Laya's own card shows the same pattern: held-out task families 0.753 → 0.651 accuracy, ECE
  0.030 → 0.204.
- Phrasing of the question matters enormously zero-shot (graph AUROC 0.57–0.83 across three
  wordings of the same question).
- Fine-tune 1 learned a shortcut: an unfamiliar-looking token means a name means graph — 17 of
  18 typo flips were false→true ("Dorughts in Califonria"). Fine-tune 2 (more varied data) halved
  the damage.
- Confidence of the fine-tunes is near-binary (98.5% of answers at ≈0.02 or ≈0.98, errors
  included), and the yes/no temperature hits the 5.0 cap: overtrained (loss ≈0 by step 100 of
  1,575). A low-confidence "escalate to the LLM" fallback will not work with these checkpoints.
- The LLM router never sends popularity queries to graph (4/4 missed) although Neo4j has a
  `by_popularity` pattern — a prompt gap in `LLM_USER_TEMPLATE` (not yet fixed).
- **But see §9: `_llm_route` only runs behind the legacy `/query` endpoint.**

## 3. Tool selection (first tool the analyze peer calls)

89 requests, 53 options (51 tools bound by `default_analyze_fn` with files attached, minus
QGIS/MCP, plus `request_capability`, plus `none`). "Lenient" also accepts inspecting an attached
file first.

| Router | strict | lenient |
|---|---|---|
| gpt-5.6-luna, single call with real tool schemas (**actual default**) | 0.562 | 0.876 |
| gpt-4o, same | 0.663 | 0.921 |
| MiniLM top-1 (no Laya) | 0.225 | 0.247 |
| Laya zero-shot (best mode) | 0.348 | 0.348 |
| Fine-tune 1 (no tool data) | 0.303 | 0.337 |
| Fine-tune 2 (tool data, v1 labels) — all 53 / shortlist 10 | 0.573 / 0.551 | 0.607 / 0.663 |

Fresh held-out set (42 requests, written after the gold results; same input format):

| Router | strict | lenient |
|---|---|---|
| gpt-4o | 0.667 | 0.952 |
| Fine-tune 2 (v1 labels) | 0.548 | 0.595 |
| **Fine-tune 3 (v2 labels) — all 53** | **0.786** | 0.786 |

Findings:
- Tool data is what matters; graph/spatial training transfers nothing to tools.
- 53 options exceed Laya's default option budget; `head_max_len` must be raised (1,400 used) or
  options shortlisted. The MiniLM shortlist keeps the gold tool only 74% of the time — a hard
  ceiling for shortlist mode.
- v1 labels taught a `request_capability` shortcut (18 of 38 misses, at P≈0.98), caused by the
  labelling prompt; the v2 prompt says which tools fetch their own inputs. Fine-tune 3 was trained
  on v2 labels **before** the data cleaning in §6.
- `detect_time_column`'s own description says to call it first for when/trend questions, so the
  models that open with it are following the tool; the gold set does not credit it.
- **But see §8: this input format is not what the pipeline passes, so these comparisons favour
  Laya.**

## 4. The real pipeline as a labeller

`scripts/pipeline_label.py` runs the local API with `AGENT_SUPERVISOR=1`, the map UI's exact
payload (Spatial on → `include_mcp_tools`, separate peers → `unifiedPeer:false`), real uploaded
stand-in files, and records per turn: supervisor decisions, every tool call with its peer, map
layers, errors. Turns whose model call failed are marked failed, never labelled.

89 gold requests through the pipeline:

| | gpt-oss:120b (AnvilGPT) | gpt-5.6-luna |
|---|---|---|
| first call, strict / lenient | 0.416 / 0.652 | 0.393 / 0.809 |
| **first substantive tool** (after inspection/`detect_time_column`) | 0.618 | **0.764** |
| same substantive tool as the other model | 64/89 = 0.719 | |
| supervisor's first peer | code 40, analyze 25, search 17 | analyze 70, search 7, code 4 |
| median seconds per turn | 192 | 57 |

Findings:
- In the product, luna almost always **looks before acting**: 40 of 89 first calls are
  `inspect_vector`/`inspect_file_for_analysis` (+8 `detect_time_column`), because the model is
  never shown column metadata (§8). "First tool" as a label mostly means "inspect".
- Luna often writes `execute_code` where a purpose-built tool exists (heat map, clusters, filter,
  nearest distance) — a prompt question for the analyze peer.
- gpt-oss is not a stand-in for luna: different supervisor routing, more listing/search detours,
  3× slower. Labels from it would teach gpt-oss's habits.
- Locally `execute_code` fails (no Docker), which makes later steps loop; the first calls are
  unaffected.

## 5. Model access notes (verified)

- **The deployed default is `gpt-5.6-luna`**, via `OPENAI_CHAT_MODEL` in `build_default_llm`;
  the `gpt-4o-2024-11-20` constant applies only when the variable is unset. AGENTS.md's "no model
  → gpt-4o" and the comment in `map-ui-prototype/src/agentClient.ts` are stale.
- **Claude subscription token (`CLAUDE_CODE_OAUTH_TOKEN`)**: Haiku 4.5 answered with tools
  (directly and through `build_llm`) for roughly the first 60 pipeline turns, then every call
  returned 400 "Third-party apps now draw from extra usage, not plan limits." Sonnet 5 / Opus 5.5
  returned 429 from the start. Not usable for batch labelling; no Haiku labels were kept.
- **AnvilGPT**: `gpt-oss:120b` works with tools; `qwen3.6:27b` now returns "Model not found".
  One transient outage ("authentication database is temporarily unreachable") was caught and
  retried.
- Secondary LLM callers (`call_llm`, search agents' `_llm_chat`, OpenGeoData) read `VLLM_*`
  before `OPENAI_*`, so pointing `VLLM_*` at AnvilGPT moves them off OpenAI without affecting a
  request that selects `provider=openai`.

## 6. Training-data review and cleaning

Automatic checks plus reading every flagged row found ~2–3% obviously broken rows.

| Problem | Rows | Action |
|---|---|---|
| Generator echoed the style instruction (`"Imperative?"`, `"Imperative: retrieve…"`) | 16 graph/spatial | dropped |
| Request refers to an uploaded file, context says none | 12 tools | dropped |
| File listing leaked into the request text | 11 tools | dropped |
| Refers to map state the format cannot carry ("the current map layer") | 4 tools | dropped |
| Duplicated `Attached files:` prefix | 2 tools | repaired |
| Copernicus / ESA named but labelled not-graph | 4 graph/spatial | relabelled |
| ~22 residual wrong `request_capability` labels; 3 imperatives labelled `none` | tools | left for pipeline relabelling |
| "Global map of…" generated as "place" but labelled not spatial | 20 | correct per rubric; kept |

Result: `train_clean.jsonl` 3,525 rows, `tools_generated_clean.jsonl` 2,834 requests
(`outputs/laya_train/clean_log.json` lists every change). Open rubric ambiguity: "resources using
<named tool>" — named software is not graph, "using a named resource" is; the labeller split 16
rows both ways.

## 7. Environment gotchas (this Windows machine)

- The Claude app's `AppData\Local` is redirected to `…\Packages\Claude_…\LocalCache\Local`; the
  first torch install failed on the 260-character path limit (the fix is a shorter path, not a
  registry change).
- Smart App Control blocks scikit-learn's compiled extension; installing it breaks `transformers`
  imports in that venv. MiniLM is loaded through `transformers` directly instead of
  sentence-transformers.
- The RTX 3060 (12 GB) needs the `cu130` torch wheel; fine-tunes took 8.5 min (graph/spatial) and
  ~75 min (tools, 53-option rows).

## 8. The input does not match the pipeline

What the analyze peer's model actually receives (`agent_chat_service._augment_user_input_with_file_ids`):

```
<request>

Uploaded files are available to the agent via local file tools. Use the exact uploaded file ids…
- file_54bf2a82ba0e (crimes.csv)
```

What Laya was trained and scored on: `Attached files: crimes.csv (points; columns: …)\nRequest: …`.

- Laya saw geometry type and columns; the real model sees only id + filename (hence inspect-first).
- Order, the fixed instruction block, the UI's drawn-region hint, search evidence, the prior-turn
  ledger and chat history are all missing from Laya's input.
- Laya always saw the same 53 options; the real bound list varies (no analysis families without
  uploads, no vector tools for CSV-only, plus QGIS/MCP/retrieval tools when present).
- The real decision is two-level (supervisor → peer → tool loop), not a flat first-tool choice.

## 9. Decisions the pipeline actually makes (map-UI path)

From code (`api/server.py` → `agent_chat_service` → `graph_runtime` → `orchestrator_graph` →
`supervisor/graph.py`) and 178 recorded turns.

| Decision | Who | Options | Fires |
|---|---|---|---|
| Triage (`orchestrator_graph.py:132`) | 2 regexes | fast / capabilities / orchestrate | once; 176/178 orchestrate |
| **Supervisor `decide()`** (`supervisor/graph.py:1798`) | LLM, per-request model | search / analyze / code / done | median 2 (luna) – 4 (oss) per turn, max 9 |
| Supervisor vetoes, caps, queued needs | rules | forced action | every step |
| Search short-circuits (related / element id / popularity) | regex | Neo4j lookup or continue | per search step |
| Search-peer retrieval | LLM ReAct | 12 retrieval tools (+QGIS/MCP/file tools) with generated args | per search step |
| Baseline sweep, web fallback | regex / rules | which backends | per search step |
| Rerank | LLM scoring | top 8 docs | per search step |
| Text2Cypher, OpenGeoData normaliser | LLM, **env model** | generated query | inside tools |
| Analyze / code peer tool calls | LLM ReAct | bound tools with generated args | per step, multi-call |
| Post-run checks (stuck tools, map not delivered, model mismatch, unrun code) | rules | one extra peer run each | conditional |
| Answer path | rules | general / insufficiency / grounded synthesis | per synthesize |
| Grounding audit | LLM | severity + claim ledger | per synthesize |
| Reground gate | rules | back to analyze/search once | 43/178 turns |

Consequences:
- `_llm_route` (graph/spatial) is reachable only via `/query` and `tool_strategy=full_pipeline`.
- Best classifier targets: `decide()` (4 labels, structured "progress so far" input, fires most),
  rerank (a cross-encoder), triage and the regex gates (already classifiers). Tool calls inside
  the ReAct peers, synthesis, Text2Cypher and the audit need generation.

Pipeline issues found in passing (code reading; the first two verified):
- `smart_tool_routing` (sent by the UI) is never read on the supervisor path.
- The appended file block breaks anchored regexes: once a session has an upload, the trivial
  fast path cannot fire; the inventory also traced the web fallback turning off
  (`_PLATFORM_HOLDINGS_RE` matches "uploaded file ids"), OpenGeoData being skipped, and the
  boilerplate reaching keyword/semantic search.
- The code peer ignores the request's `enabledSearchMethods` (hardcoded allowlist).
- Text2Cypher, the OpenGeoData normaliser and the search peer's rerank/audit tools ignore the
  per-request model.

## 10. Open decisions / next steps

1. Pick the Laya target from §9 — recommended `decide()`; the recorded traces already contain
   its inputs and outputs.
2. For whatever target, build inputs with the pipeline's own functions (e.g.
   `_augment_user_input_with_file_ids`, `_distill`) and the per-turn bound option list.
3. Labels: gpt-5.6-luna through the pipeline matches production (cost: several calls per turn);
   gpt-oss is free but a different, weaker target (§4).
4. Fix calibration before relying on confidence: fewer epochs / early stopping, label smoothing.
5. Build and run the sidecar image on a machine with Docker.
6. The pipeline issues in §9 are worth fixing independently of Laya.

## Reproducing

```bash
# graph/spatial
PYTHONPATH="$PWD" python scripts/eval_laya_router.py llm --out outputs/router_eval/llm.json
PYTHONPATH="$PWD" python scripts/eval_laya_router.py laya --url http://127.0.0.1:8010 --variant short --out outputs/router_eval/laya_short.json
PYTHONPATH="$PWD" python scripts/make_router_shifts.py --paraphrase
PYTHONPATH="$PWD" python scripts/router_shift_report.py laya-ft-tools

# tool selection
PYTHONPATH="$PWD" python scripts/eval_laya_tools.py llm --model gpt-5.6-luna --effort none --out outputs/tool_eval/llm_gpt-5.6-luna.json
python scripts/eval_laya_tools.py laya --model-dir <checkpoint> --name ft --mode full --out outputs/tool_eval/laya_ft_full.json
python scripts/eval_laya_tools.py report outputs/tool_eval/*.json      # add --gold scripts/tool_eval_holdout.jsonl for the held-out set

# training data and fine-tuning
python laya-server/make_training_data.py generate --out outputs/laya_train/generated.jsonl
python laya-server/make_tool_training_data.py generate --out outputs/laya_train/tools_generated.jsonl
python laya-server/clean_training_data.py
python laya-server/finetune.py --base <laya dir> --train outputs/laya_train/train_clean.jsonl --out <out dir>

# real-pipeline labelling (local API on :5055, AGENT_SUPERVISOR=1)
python scripts/pipeline_label.py --inp scripts/tool_eval_queries.jsonl --out outputs/pipeline_label/gold_luna.jsonl --provider openai --model gpt-5.6-luna
```
