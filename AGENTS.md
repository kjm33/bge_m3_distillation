# AGENTS.md

Guidance for AI coding agents working in this repo.

## Project

Distillation of **BAAI/bge-m3** (teacher) into a CPU-deployable embedding model
for car-navigation instructions (en/de/fr/pl/es), plus a quantization/vocab-
pruning ladder. Deployed pick: `bge_m3/models/edge/nav_e5s_distill_vocab4s10.onnx`
+ `nav_tok10/` pruned tokenizer (14.49 MB, sp 0.8945 vs teacher). Full context:
`README.md` (reproduction), `bge_m3/RESULTS.md` (lab notebook),
`bge_m3/PROGRESS.md` (summary/decisions). Read `bge_m3/PROGRESS.md` before
changing anything in the pipeline — most "obvious" ideas were already tried
and rejected (see its "Decisions & rejected alternatives" section).

## Commands

All pipeline scripts run **from `bge_m3/`** (paths are relative to it), with
the repo-local venv:

```bash
cd bge_m3
.venv/bin/python train_distill.py --stage teacher|train|onnx|all
.venv/bin/python quant_ladder.py --stage baseline|int4|vocab|vocab4|static|tokprune|vocab4e|vocab4s|fresh_audit|all
.venv/bin/python edge_compare.py --stage quality|cpu|onnx|all
```

There is **no test suite, no linter, no typechecker** configured. Verification
is the pipeline's built-in gates (run the relevant stage and check its output):
vocab-prune sanity gate cos ≥ 0.9999 vs torch, per-rung sp floor 0.89 + <15 MB
budget verdicts, pruned-tokenizer id-identity on corpus+bench. Result JSONs
land in `bge_m3/data/*_results.json`. After a rerun you only wanted side
effects from, restore tracked results with
`git checkout -- bge_m3/data/`.

## Environment

- Python 3.12 venv at `bge_m3/.venv`; pinned versions in `README.md`.
- Scripts set `HF_HUB_OFFLINE=1` by default — HF weights must be pre-cached
  (`hf download BAAI/bge-m3`, `intfloat/multilingual-e5-small`) or run with
  `HF_HUB_OFFLINE=0` once.
- GPU (CUDA) required for teacher/GT/training stages; the quantization ladder
  and ONNX benching are CPU.
- Determinism: seed 42 throughout; metrics quotes in docs assume it.

## Gotchas

- **Corpus is vendored**: `data/instructions_v3.jsonl` (21,959 rows, 15 MB) is
  the raw source corpus; `SRC` in all scripts points at it in-repo. Derived
  files (`data/train_texts.jsonl`, `data/edge_bench.jsonl`,
  `data/bench_texts.json`, `data/nav_token_freq.json`) were built from it.
- **Tracked caches silently reused**: `data/train_texts.jsonl`,
  `data/nav_token_freq.json`, `data/bench_texts.json` are inputs-or-caches —
  delete the relevant ones when changing datasets, or stale results will be
  reused instead of recomputed.
- **Gitignored by design**: `bge_m3/models/` and `data/*.npy` /
  `data/embeddings_cache.db` are regenerable (see README steps) — never commit
  them; also never delete a colleague's local `models/` casually (GPU rebuild
  costs ~40 min).
- `data/edge_gt_fp32.npy` (bench ground truth) is gitignored and required by
  both `train_distill.py` and `quant_ladder.py`; rebuild via the snippet in
  README §"Rebuild the bench ground truth".
- Dataset JSONL format: `{"text": <verbatim string>, "language": <code>}` —
  see README "Dataset format".

## Conventions

- Scripts are single-file, stdlib + few deps, heavy docstring headers
  describing stages — keep that pattern; new pipeline steps become a new
  `--stage` in the relevant script, not a new script.
- Numbers in docs cite the JSON in `data/`; when a change moves a metric,
  update `PROGRESS.md`/`RESULTS.md` and the result JSON in the same commit.
- Commit message style: `bge_m3: <what and why>` (see `git log`).
- Embedding outputs are L2-normalized; student uses mean-pooling, teacher CLS;
  all texts get the `"query: "` e5 prefix — preserve in any new encode path.
