# embeddings — distilled edge embedding model for navigation instructions

Distills **BAAI/bge-m3** (teacher, 2.3 GB server model) into a CPU-deployable
embedding model for car-navigation instructions in 5 languages (en/de/fr/pl/es),
then shrinks it down a quantization/vocab-pruning ladder to **14.49 MB** at
~90% of the teacher's ranking fidelity. Everything is reproducible from this
repo: corpora, bench sets, token-frequency census and result JSONs are tracked;
model weights and embedding arrays are gitignored by design and rebuilt by the
scripts below.

Full methodology: `bge_m3/RESULTS.md` (chronological lab notebook),
`bge_m3/PROGRESS.md` (summary + decisions).

## The deployed pick

| tier | artifacts (under `bge_m3/models/edge/`) | total size | sp 20k | p50 |
|---|---|---|---|---|
| conservative | `nav_e5s_distill_vocab8.onnx` + full tokenizer + LUT | 43.7 MB | 0.896 | 4.3 ms |
| max quality ≤16 MB | `nav_e5s_distill_vocab4e.onnx` + `nav_tok15/` | 15.65 MB | 0.901 | 5.3 ms |
| **current pick (<15 MB)** | **`nav_e5s_distill_vocab4s10.onnx` + `nav_tok10/`** | **14.49 MB** | **0.895** | **5.1 ms** |

sp = Spearman correlation vs BGE-M3 fp32 teacher over 20k random pairs of the
512 held-out bench texts (seed 42). Reference: teacher int8 ONNX is 542 MB,
p50 41 ms.

## Repo layout

```
bge_m3/
  train_distill.py     # §7: teacher embedding -> distillation training -> ONNX export
  quant_ladder.py      # §8/§8.1: int4/vocab-pruning ladder -> vocab4s rungs (the pick)
  edge_compare.py      # §6: stock-model screen + bench GT builder (needs corpus at SRC)
  data/                # tracked inputs & results (train_texts.jsonl, edge_bench.jsonl,
                       #   nav_token_freq.json, *_results.json)
  models/              # gitignored output (rebuilt by the scripts)
  notes/               # long-form explanations
```

## Requirements

- **Hardware**: NVIDIA GPU with CUDA for the teacher/GT/training steps
  (reference machine: RTX 3090 24 GB; training takes ~25 min). The
  quantization ladder and inference are CPU-only.
- **Software**: Python 3.12 + pinned deps (versions used for the shipped
  results):

```bash
python -m venv .venv
.venv/bin/pip install torch==2.14.0 transformers==5.17.0 sentence-transformers==6.1.0 \
    onnx==1.23.0 onnxruntime==1.30.0 tokenizers==0.23.2 scipy==1.18.1 \
    numpy==2.5.3 psutil==7.2.2 huggingface_hub==1.33.0
```

(Install `torch` from the CUDA 13.0 wheel index if your default pip resolves a
CPU build: `pip install torch --index-url https://download.pytorch.org/whl/cu130`.)

## Warm the HuggingFace cache

Both pipeline scripts default to `HF_HUB_OFFLINE=1`, so the teacher/student
weights must already be in the local HF cache. Either pre-download:

```bash
.venv/bin/hf download BAAI/bge-m3              # older hub: huggingface-cli download ...
.venv/bin/hf download intfloat/multilingual-e5-small
```

…or run the first step with network allowed: `HF_HUB_OFFLINE=0 .venv/bin/python ...`.

## Dataset format

Two JSONL files drive the pipeline, one object per line:

```json
{"text": "Turn right onto Main Street", "language": "en"}
```

- `text` (required): non-empty string, used **verbatim** (raw JSON-artifact
  strings are fine — they are embedded as-is).
- `language` (optional): used only for per-language metric splits
  (mono vs cross-lingual); defaults to `"?"`.

| file | role | size |
|---|---|---|
| `bge_m3/data/train_texts.jsonl` | training pool (bench overlap + dupes already excluded) | 21,447 lines |
| `bge_m3/data/edge_bench.jsonl` | held-out eval bench (512 texts) | 512 lines |

## Reproduction steps

Run everything from `bge_m3/`:

```bash
cd bge_m3
```

### 1. Rebuild the bench ground truth (GPU, ~1 min)

`data/edge_gt_fp32.npy` (teacher embeddings of the 512 bench texts) is
gitignored. Rebuild it with BGE-M3 fp32 CUDA CLS pooling — save as
`make_gt.py` and run `.venv/bin/python make_gt.py`:

```python
import json, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
texts = [json.loads(l)["text"] for l in open("data/edge_bench.jsonl") if l.strip()]
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3").to("cuda").eval()
out = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i+64], padding=True, truncation=True,
                  max_length=8192, return_tensors="pt").to("cuda")
        out.append(model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
v = np.concatenate(out).astype(np.float32)
v /= np.linalg.norm(v, axis=1, keepdims=True)
np.save("data/edge_gt_fp32.npy", v)
print(v.shape)  # expect (512, 1024)
```

(Identical method to `build_gt()` in `edge_compare.py`; that script's full
stock-model screen additionally needs the source corpus at its `SRC` path and
is not required to reproduce the pick.)

### 2. Distill + export (GPU, ~30–40 min total)

```bash
.venv/bin/python train_distill.py --stage all
```

Stages: `teacher` (embed the pool with BGE-M3 fp32 CUDA), `train`
(hard-neighbor KL-matrix distillation into multilingual-e5-small, Matryoshka
{64,128,256,384} dims, 8 epochs, best checkpoint per bench eval), `onnx`
(fp32 + int8 dynamic export, CPU bench). Reuses the tracked
`data/train_texts.jsonl` — no external corpus needed.

Outputs: `models/nav-e5s-distill/` (student checkpoint),
`models/edge/nav_e5s_distill.onnx` + `nav_e5s_distill_int8.onnx`,
results merged into `data/distill_results.json`.

### 3. Build the deployed artifact (CPU, minutes)

```bash
.venv/bin/python quant_ladder.py --stage vocab4s
```

Builds the int4 rungs (int4-packed word/pos tables + MatMulNBits body) with
kept-vocab capped at 12k/10k/8k using the tracked `data/nav_token_freq.json`
census — no external corpus needed. The sweep prints per-rung size, Spearman,
and the sp-floor 0.89 / 15 MB budget verdicts; **vocab4s10** is the pick.

Outputs: `models/edge/nav_e5s_distill_vocab4s{12,10,8}.onnx`,
`models/edge/nav_tok{12,10,8}/` (pruned tokenizers),
`models/edge/nav_vocab_kept_ids_{12,10,8}k.json`, results merged into
`data/quant_ladder_results.json`.

### 4. Optional: fresh-query audit vs live teacher (GPU)

```bash
.venv/bin/python quant_ladder.py --stage fresh_audit
```

10 hand-written fresh queries through every vocab4* artifact on the deployment
path, verdicts vs live BGE-M3 fp32 CUDA.

### Verification numbers (tracked dataset)

| step | metric | expected |
|---|---|---|
| 1 | GT shape | (512, 1024) |
| 2 | student fp32 sp / int8 ONNX | 0.908 / 0.897 (int8 112.6 MB, p50 4.7 ms) |
| 3 | vocab4s12 / **s10** / s8 | 14.95 MB sp 0.8954 / **14.49 MB sp 0.8945** / 14.04 MB sp 0.8882 |

Note: rerunning updates the *tracked* result JSONs (`data/distill_results.json`,
`data/quant_ladder_results.json`) — `git checkout -- bge_m3/data/` restores
them if you only wanted the artifacts.

## Inference contract (vocab4s10)

- Prefix every input with `"query: "` (e5 convention, all languages).
- Pruned tokenizer (`models/edge/nav_tok10/tokenizer.json`): `no_padding()`,
  truncation to 256 tokens (position table hard-limits at 320).
- Pad id `1`, mean-pool the (B, L, 384) output with the attention mask, L2-normalize.
- No LUT/remap needed — the pruned tokenizer emits new ids natively.

```python
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

tok = Tokenizer.from_file("models/edge/nav_tok10/tokenizer.json")
tok.no_padding(); tok.enable_truncation(max_length=256)
sess = ort.InferenceSession("models/edge/nav_e5s_distill_vocab4s10.onnx",
                            providers=["CPUExecutionProvider"])
texts = ["Turn right after the bridge", "Nach der Brücke rechts abbiegen"]
ids = [e.ids for e in tok.encode_batch(["query: " + t for t in texts])]
maxlen = max(len(x) for x in ids)
input_ids = np.full((len(ids), maxlen), 1, dtype=np.int64)      # pad id = 1
mask = np.zeros((len(ids), maxlen), dtype=np.int64)
for j, x in enumerate(ids):
    input_ids[j, :len(x)] = x; mask[j, :len(x)] = 1
feed = {"input_ids": input_ids, "attention_mask": mask}
if "token_type_ids" in [i.name for i in sess.get_inputs()]:
    feed["token_type_ids"] = np.zeros_like(input_ids)
h = next(o for o in sess.run(None, feed) if o.ndim == 3)
m = mask.astype(np.float32)
emb = (h * m[:, :, None]).sum(1) / np.maximum(m.sum(1)[:, None], 1.0)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)               # (B, 384), unit norm
```

## Fine-tuning on your own dataset

The pipeline is dataset-agnostic given the JSONL format above.

1. Put your corpus at a path and point `SRC` at it in `quant_ladder.py:51`
   (`train_distill.py:35` / `edge_compare.py:42` likewise if you want the
   pool rebuilt from source rather than from a hand-written
   `train_texts.jsonl`).
2. Replace the eval bench `data/edge_bench.jsonl` with ~512 held-out texts
   from your domain, then rerun step 1 (GT rebuild).
3. Delete derived caches so they regenerate instead of being reused:
   `data/train_texts.jsonl`, `data/teacher_full_bge_m3.npy`,
   `data/nav_token_freq.json`, `data/bench_texts.json`.
4. Rerun steps 2–3. The token-frequency census and kept-vocab ids recompute
   for your corpus automatically.

Expect different absolute metrics on a new domain: the sp floor (0.89) and
15 MB budget checks printed per rung are the re-evaluation criteria for which
rung ships; vocab4s10 is not guaranteed to be the right pick for a different
corpus.
