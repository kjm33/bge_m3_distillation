# BGE-M3 — OpenRouter determinism, caching, local resources & int8 quantization

Date: 2026-09-26 · Hardware: Ryzen 9 5950X (16C/32T), 62 GB RAM, RTX 3090 24 GB · Python 3.12, torch 2.14+cu130, onnxruntime 1.30, sentence-transformers 6.1

OpenRouter model: `baai/bge-m3` — $0.01/1M tokens, context 8194, 1024-dim L2-normalized vectors, one vector per input string (CLS-pooled dense; verified: API↔local CLS cosine 0.9989–0.9999, mean pooling 0.70–0.78 → rejected). Served by DeepInfra (fp32) and Parasail (quantization "unknown"); neither supports implicit caching.

## 1. Determinism of OpenRouter embeddings — "static, not bit-exact"

Docs claim "always identical". Empirically (15 real multilingual instructions + edge cases):

| Test | Result |
|---|---|
| Same batch ×5 consecutive | runs flip between **≥2 serving backends**: within a backend bit-exact; cross-backend drift **max|Δ| ≈ 1.5–3.5e-4, cosine ≥ 0.999998** (all texts flip together → per-request backend routing) |
| Delayed retry (20 s) | same noise level (≤ 2.8e-4) |
| Batch vs single-item requests | ≤ 3.0e-4, cosine ≥ 0.999998 |
| Pinned DeepInfra vs pinned Parasail | every text differs ≤ 4.7e-4, cosine ≥ 0.9999988 — providers near-identical but not bit-identical |
| Duplicate texts inside one batch | bit-exact (probe; one earlier non-exact case = same cross-backend noise) |
| API vs local fp32 PyTorch (CLS) | cosine 0.9989–0.9999 → API ≈ fp32 dense |

**Verdict:** functionally static — differences live in the 7th significant digit and are invisible to retrieval/dedup/clustering. NOT bit-exact across requests/providers, so OpenRouter's "identical" claim holds only per-backend. Test cost: ~14k tokens ≈ $0.0001.

## 2. Local caching — safe, mandatory, 1343× faster

Server does no caching (`supports_implicit_caching: false` on both providers) → every repeat request recomputes & re-bills. Client-side cache (`embed_cache.py`, SQLite):

- Key: `sha256(model | provider | NFC+strip(text))` — provider in key for hygiene; numerically optional given §1.
- Value: fp32 blob, 4 KB/vector (L2-normalized ⇒ cosine-ready). DB overhead → 4.77 KB/vector actual.
- Benchmark (200 real instructions): cold **9.35 s (46.7 ms/text)** → warm **7 ms (0.035 ms/text)** = **1343× speedup**; single lookup ~3.3 ms.
- Space-constrained option — store int8 + per-vector scale (1 KB + 4 B): min cosine 0.99978 vs fp32, top-1 NN agreement 98%.
- Invalidation: only on model change (model id already in key).

## 3. Local resource profile (512 real instructions, ~23 tok avg)

Disk: fp32 weights **2.12 GB** (pytorch_model.bin) + official ONNX **2.13 GB** (HF cache total 12.77 GB incl. duplicates/sparse/colbert heads).

| Config | Load | Peak RAM | VRAM | bs1 texts/s | bs32 texts/s | p50 latency | 10.8k-token doc |
|---|---|---|---|---|---|---|---|
| CPU fp32 (16 threads) | 0.6 s | ~2.5 GB | – | 4.7 | 19.8 | 181 ms | 577 tok/s |
| GPU fp32 | 1.2 s | ~1.8 GB | 2.9 GB | 87 | 452 | 11 ms | 9.7k tok/s |
| GPU fp16 | 1.1 s | ~2.0 GB | **1.5 GB** | 85 | **1378** | 12 ms | **53k tok/s** |

## 4. Dynamic int8 ONNX quantization (CPU)

`quantize_dynamic` on the official ONNX export (external-data format). Quantize time 17 s.

| Metric | fp32 ONNX | int8 ONNX | Δ |
|---|---|---|---|
| Size | 2.11 GB | **542 MB** | **3.99×** |
| Session load | 1.37 s | 0.40 s | 3.4× |
| Throughput bs8 | 431 tok/s | **702 tok/s** | 1.63× |
| Single-text p50/p95 | 99.6 / 117.8 ms | **41.0 / 52.6 ms** | 2.43× |

Quality vs PyTorch fp32 GPU ground truth (CLS, normalized):

| Metric | fp32 ONNX (sanity) | int8 ONNX |
|---|---|---|
| cosine mean / min | 1.0 / 1.0 | 0.9806 / 0.9647 |
| max coord diff | 1e-6 | 0.035 |
| Pairwise-sim Spearman (20k pairs) | 1.0 | 0.978 |
| Retrieval Jaccard@10 | 1.0 | 0.778 |
| MRR of GT top-1 | 1.0 | 0.907 |

Official ONNX export matches PyTorch **exactly** (cos 1.0). int8 loses ~2% cosine and reshuffles ~22% of top-10 neighbors — fine for dedup/clustering/prefilter, risky for precision-critical retrieval. For quality-preserving slimming prefer **GPU fp16** (1.5 GB VRAM, ~fp32 quality, 31k tok/s).

## 5. Recommendations

1. **Cache OpenRouter results locally** — 1343× cheaper/faster; never re-embed static corpora. Pin a provider in production requests.
2. **Local default:** GPU fp16. CPU-only: int8 ONNX (2.4× latency win, 4× disk win) when quality bar is dedup-grade; keep fp32 ONNX for exact reproduction.
3. Comparison point: local GPU fp16 (1378 texts/s ≈ 1.4M instructions/h) vs API (21 texts/s observed) — local is ~65× faster and cost-free after the 2.3 GB download.

## 6. Edge-device model comparison (quality vs BGE-M3, CPU cost, ONNX int8, Matryoshka)

Bench: 512 natural instructions (5 langs balanced; 880 JSON artifacts filtered; ~33 tok/text). Ground truth: BGE-M3 fp32 CUDA CLS (§3 pipeline). Excluded upfront: jina-embeddings-v3 (CC-BY-NC license), nomic-embed-text-v2-moe (custom `NomicBertModel` incompatible with transformers 5.17 — needs transformers<5), OpenRouter (no embedding models besides bge-m3). Full data: `data/edge_compare_results.json`.

### 6.1 Quality vs BGE-M3 ground truth

| model | params | dim | spear↑ | jac@10↑ | MRR@10↑ | top1↑ | top1 **xl**↑ |
|---|---|---|---|---|---|---|---|
| e5_base | 278M | 768 | 0.571 | **0.388** | **0.598** | **0.478** | 0.059 |
| labse | 471M† | 768 | **0.725** | 0.350 | 0.529 | 0.383 | **0.309** |
| e5_small | 118M | 384 | 0.572 | 0.356 | 0.551 | 0.424 | 0.012 |
| minilm | 118M | 384 | 0.471 | 0.295 | 0.446 | 0.316 | 0.202 |
| potion (static M2V) | 128M | 256 | 0.501 | 0.303 | 0.472 | 0.350 | 0.036 |
| qwen3-emb-0.6B | 596M | 1024 | 0.420 | 0.365 | 0.543 | 0.422 | 0.095 |

† LaBSE is 471M params (501k-token vocab embedding dominates), fp32 weights 1.75 GB — heavier than it looks.

**Headline finding: agreement with BGE-M3 is low across the board.** Best pairwise-Spearman 0.725 (labse), best Jaccard@10 only 0.388 (e5_base) — every model reshuffles most of the top-10 neighborhood. Size buys no agreement: Qwen3-0.6B (596M) scores *lower* than e5_small (118M, 0.572 vs 0.420). Interpretation: on this narrow navigation domain, BGE-M3's neighborhood structure is idiosyncratic; **swapping the embedding model materially changes retrieval results**, so "cheaper drop-in replacement for BGE-M3" does not exist — a replacement must be re-validated on the actual task, or distilled (→ TODO item 3). Only LaBSE reproduces BGE-M3's cross-lingual alignment (top1xl 0.309 vs ≤0.20 elsewhere; e5_small effectively none at 0.012).

### 6.2 CPU fp32 cost (16 threads)

| model | bs32 texts/s | p50 ms | load s | ΔRSS MB |
|---|---|---|---|---|
| potion | 29 211 | 0.2 | 2.1 | 527 |
| e5_small | 244 | 21.7 | 17.4* | 289 |
| minilm | 236 | 20.9 | 19.5* | 203 |
| labse | 88 | 60.7 | 0.8 | 527 |
| e5_base | 76 | 64.5 | 26.6* | 881 |
| bge_m3 (ref) | 25 | 234.0 | 1.7 | 1374 |
| qwen3-0.6B | 3.7 | 144.4 | 0.6 | 972 |

\* first-load sentence-transformers conversion cost; warm loads ~1 s. Qwen3-0.6B is memory-bound on CPU and *slows down* when batching (6.0→3.7 texts/s) — not edge-viable; it's the GPU/MRL option instead.

### 6.3 ONNX dynamic int8 (official exports, CPU)

| model | int8 size | bs8 texts/s | p50 ms | cos vs fp32-torch | Jaccard@10 vs GT |
|---|---|---|---|---|---|
| minilm | 112.7 MB | 241 | **3.9** | 0.987 | 0.294 |
| e5_small | 112.6 MB | 230 | 5.1 | 0.988 | 0.345 |
| e5_base | 265.3 MB | 83 | 12.8 | 0.975 | 0.360 |
| labse | 449.4 MB | 66 | 51.2 | **0.520** ⚠ | 0.316 |

- 3.98–3.99× size reduction everywhere; p50 latency 4.3–5.4× faster than PyTorch fp32 CPU (e.g. minilm 20.9→3.9 ms, e5_base 64.5→12.8 ms).
- **int8 breaks LaBSE** (cosine 0.52 vs fp32 — sensitive LayerNorm/vocab dynamics) though rank metrics survive ≈; if LaBSE is chosen, keep fp32/fp16.
- e5_small int8 ONNX ≈ **7.5k tok/s** at 113 MB — the best quality-per-byte CPU package here (vs BGE-M3 int8: 702 tok/s, 542 MB, p50 41 ms).

### 6.4 Matryoshka / dimension truncation

| model | MRL-trained | dims | spearman vs GT by dim |
|---|---|---|---|
| qwen3-0.6B | yes | 64/128/256/512/1024 | 0.427 / **0.486** / 0.457 / 0.420 / 0.420 |
| bge-m3 (naive) | no | 128/256/512/1024 | 0.807 / 0.905 / 0.968 / 1.000 |
| e5_small (naive ctrl) | no | 96/192/384 | 0.495 / 0.551 / 0.572 |

- Qwen3 MRL truncation to **128d improves** GT-agreement over its full 1024d (0.486 vs 0.420) — the tail dims carry noise for this domain; MRL dims are importance-ordered and 128d retains the signal.
- BGE-M3 naive truncation is a usable freebie: 512d keeps Spearman 0.968 / Jaccard@10 0.734 (halve storage for caches/indexes with ~3% rank loss); 256d still 0.905/0.595.

### 6.5 Edge recommendations

1. **Default CPU edge model: e5_small (or minilm) int8 ONNX** — ~113 MB, p50 4–5 ms, 230–240 texts/s; accepts that neighbor sets differ from BGE-M3 (jac@10 0.35 vs GT).
2. **Must match BGE-M3 cross-lingual behavior** → only LaBSE comes close (top1xl 0.309) but it's 471M/1.75 GB and int8-hostile; or keep BGE-M3 int8 (542 MB, p50 41 ms) as the behavior-preserving option.
3. **Sub-ms tier:** potion static embeddings (29k texts/s, 0.2 ms) for prefilter/dedup only — weakest structure agreement, near-zero cross-lingual.
4. No drop-in BGE-M3 clone exists at any size → distillation (TODO item 3, recipe in `BASE_MODEL_RESEARCH.md`) is the path to small + faithful.

## 7. Training our own edge embeddings — bge-m3 → e5-small distillation

Fast path from `BASE_MODEL_RESEARCH.md`: distill BGE-M3 (teacher) into `multilingual-e5-small` (student, 118M/384d) on our own corpus (`train_distill.py`). Method: symmetric KL between teacher and student batch cosine-similarity matrices (τ=0.03), Matryoshka over dims {64,128,256,384}, hard-neighbor batches (anchor + 7 random-of-top-64 teacher NNs, 8 groups → bs 64), 8 epochs × 21,447 train texts (bench's 512 held out), lr 2e-5, bf16, 25 min on the 3090. Best epoch 6.

### 7.1 Quality vs BGE-M3 GT (512 held-out bench texts)

| metric | stock e5-small | **nav-e5s-distill** | Δ |
|---|---|---|---|
| pairwise Spearman (20k) | 0.572 | **0.908** | +0.336 |
| kNN Jaccard@10 | 0.356 | **0.623** | +0.267 |
| MRR@10 of GT top-1 | 0.551 | **0.796** | +0.245 |
| top-1 agreement | 0.424 | **0.678** | +0.254 |
| **top-1 cross-lingual** | 0.012 | **0.548** | **+0.536** |
| ARI-50 clusters | 0.252 | **0.369** | +0.117 |

The distilled 118M model beats every stock model in §6 — including 4× larger qwen3-0.6B (sp 0.420) and 471M LaBSE (sp 0.725) — and is the *only* small model with real cross-lingual alignment (0.548; LaBSE 0.309, everything else ≤0.095). Trains in 25 min on one GPU; re-runnable when the corpus grows.

### 7.2 Edge deployment (CPU, ONNX)

| variant | size | bs8 texts/s | p50 ms | cos vs torch | Jaccard@10 vs GT |
|---|---|---|---|---|---|
| stock e5-small int8 (§6) | 112.6 MB | 230 | 5.1 | 0.988 | 0.345 |
| **nav-e5s-distill int8** | 112.6 MB | 212 | 4.7 | 0.984 | **0.599** |
| nav-e5s-distill fp32 | 448.4 MB | 155 | 7.8 | 1.000 | 0.623 |

Identical cost to stock e5-small int8 (same architecture → same 112.6 MB, same speed), 74% higher GT agreement. int8 costs only sp 0.908→0.896 / jac 0.623→0.599 — quantization-tolerant (unlike LaBSE). Matryoshka-trained: truncated dims stay ordered (deploy 128–256d to halve vector storage again if needed).

### 7.3 Conclusions

1. **Deploy `models/nav-e5s-distill` int8 ONNX** as the edge model: BGE-M3-faithful (sp 0.90, cross-lingual top-1 0.55) at 113 MB / 4.7 ms — replaces the §6.5 compromise between size and fidelity. *(Superseded by §8: the vocab-pruned `vocab8` build is 26.6 MB at identical quality.)*
2. Distillation beat every alternative tried across §4/§6: quality-per-byte and per-watt champion.
3. Ceiling note: agreement tops out ~0.91 (8 epochs) — a EuroBERT-210m student or a Qwen3 ensemble teacher (see `BASE_MODEL_RESEARCH.md`) is the next lever if >0.95 is ever needed.

## 8. Quantizing further: the size ladder (vocab pruning, int4, static)

Lever discovery: nav-e5s-distill is 131.4M params of which **96M (73%) is the embedding table**, and the corpus uses only **15,199 of 250,002 vocab tokens**. So layer bit-width is a minor lever — **vocabulary pruning is the big one**. Rungs (`quant_ladder.py`, GT = same 512-text bench):

| rung | recipe | size | p50 ms | bs8 texts/s | ΔRSS MB | cos vs torch | sp 20k | jac@10 | top1 | top1-xl | ARI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| int8 (= §7 baseline) | dynamic int8 | 112.6 MB | 4.66 | 212 | 94 | 0.984 | 0.896 | 0.599 | 0.664 | 0.500 | 0.370 |
| int4 | MatMul int4 (block 32) + emb int8 | 103.8 MB | 5.14 | 144 | 187 | 0.980 | 0.901 | 0.611 | 0.678 | 0.512 | 0.308 |
| **vocab8** | vocab 250,002→**15,276** rows + pure int8 | **26.6 MB** | 4.33 | 224 | 53 | 0.984 | 0.896 | 0.599 | 0.664 | 0.500 | 0.370 |
| **vocab4** | vocab-pruned + int4 MatMul | **17.8 MB** | 5.22 | 143 | 88 | 0.980 | 0.901 | 0.611 | 0.678 | 0.512 | 0.308 |
| static | model2vec (mean of transformer outputs, PCA-300 whiten, int8) | 14.9 MB | **0.15** | 18,244 | 110 | – | 0.468 | 0.247 | 0.283 | 0.000 | 0.204 |

Findings:

1. **vocab8 is a free lunch: 4.2× smaller (112.6→26.6 MB) with bit-identical quality to the §7 baseline** (sp 0.896, jac 0.599 — same numbers, kept embedding rows are unchanged; sanity cos 1.0 vs unpruned on covered texts). Same latency (p50 4.3 ms), lower RAM (53 MB). Deployment needs the id remap (250k→15k LUT, shipped as `models/edge/nav_vocab_kept_ids.json`; tokenizer unchanged).
2. **vocab4 = 17.8 MB at sp 0.901** — slightly *above* int8 (noise-level, consistent across int4 rungs) but ~30% slower (int4 dequant overhead on this small model). Prefer vocab8 unless every MB counts.
3. Plain int4 (103.8 MB) is dominated — embedding table stays int8, so it saves only 9 MB and costs latency.
4. **Static (model2vec-style) works after domain distillation but stays prefilter-tier**: sp 0.468 ≈ stock potion (0.501) territory, cross-lingual 0. PCA-whitened 300d, mean over transformer outputs at interior positions ([bos,i,eos]). Sub-ms (0.15 ms) and 80× throughput — usable for dedup/prefilter only.
5. Trap documented: the static model inherited the e5 tokenizer's **batch padding config**; model2vec's raw-backend `tokenize` then pads batches and means over pad rows → garbage on 98% of texts (sp -0.10). Fix: `no_padding()` before save/encode (`tokenizer.json` patched on disk; both `build_static_model` and `static_embed` call it). Verified: batched encode == manual pooling, sp 0.4675.

**Updated deployment recommendation (supersedes §7.3-1):** ship **vocab8** — 26.6 MB, sp 0.896, cross-lingual top-1 0.50, 4.3 ms. That is 4.2× smaller than §7's pick at identical quality, and 20× smaller / 9.5× faster than bge-m3 int8 at ~90% rank fidelity. Combined with MRL 128d vectors (§7), index storage halves again (1.5 KB→512 B/text).

### 8.1 The <15 MB question: total shipped footprint (onnx + tokenizer)

The §8 table counted only the `.onnx` file. Real deployment ships the tokenizer too, and the full `tokenizer.json` is **17.0 MB** (250k-piece Unigram): vocab8's true footprint is 26.6 + 17.0 + 0.1 (LUT) = **43.7 MB**, vocab4's is 34.9 MB. Two new levers close the gap (`quant_ladder.py` stages `tokprune`/`vocab4e`/`vocab4s`/`fresh_audit`, results in `data/quant_ladder_results.json`):

1. **Pruned tokenizer (nav_tok*)**: filter `tokenizer.json` to the kept pieces in ascending-id order, so it *emits new ids natively* — the 250k→15k LUT is no longer needed at runtime. Hard gate: ids byte-identical to full+LUT on **0/21,959 corpus and 0/512 bench mismatches** (Unigram argmax is unchanged when every piece of the optimal path survives). Size: **0.75 MB** (15,276 pieces) down to 0.54 MB (8k). On fresh text, dropped pieces decompose into kept subwords/chars instead of mapping to `<unk>` — strictly better than the LUT path (demo queries: unk 3→0).
2. **int4 embedding tables + pos trim**: the word/position Gather tables are quantized to int4 (uint8 nibble pairs + per-row fp32 scales, exact symmetric per-row, decoded in-graph with plain opset-17 ops — no int4 dtype, no LUT), and the position table is sliced 514→320 rows (MAX_LEN 256 + headroom). Matmuls stay MatMulNBits int4 (block 64). Table cost: 15,276×384 → **2.9 MB**, 10k → 1.9 MB, 8k → 1.5 MB.

| rung | onnx | tokenizer | **total** | p50 ms | sp 20k | jac@10 | top1 | top1-xl | verdict |
|---|---|---|---|---|---|---|---|---|---|
| vocab8 (§8) | 26.6 MB | 17.0 MB + LUT | 43.7 MB | 4.33 | 0.896 | 0.599 | 0.664 | 0.500 | prev. pick |
| vocab4 (§8) | 17.8 MB | 17.0 MB + LUT | 34.9 MB | 5.22 | 0.901 | 0.611 | 0.678 | 0.512 | |
| vocab4e | 14.9 MB | 0.75 MB | **15.65 MB** | 5.33 | 0.901 | 0.607 | 0.676 | 0.512 | int4-tier, misses by 0.65 |
| vocab4s12 | 14.3 MB | 0.65 MB | **14.95 MB** | 5.40 | 0.895 | 0.605 | 0.664 | 0.476 | ✓ floor, ✓ budget (hairline) |
| **vocab4s10** | 13.9 MB | 0.59 MB | **14.49 MB** | 5.11 | 0.895 | 0.603 | 0.664 | 0.464 | **✓ floor, ✓ budget (pick)** |
| vocab4s8 | 13.5 MB | 0.54 MB | 14.04 MB | 5.37 | 0.888 | 0.600 | 0.652 | 0.429 | ✗ sp floor 0.89 |

Acceptance floor sp ≥ 0.89 (§7 harness): 12k/10k pass, 8k misses by 0.002. The body dominates (≈11.5 MB of int4 matmuls) — further shrink needs layer-drop + re-distill, not vocab.

**Fresh-query audit** (`--stage fresh_audit`, same 10 hand-written queries as the discrepancy demo; teacher live fp32 CUDA, cached): vocab8 (deployment path incl. LUT): 5 EXACT / 5 REORDER / 0 DIVERGE, drift vs int8 query emb ≥ 0.904. All int4 rungs incl. vocab4s10: 5 EXACT / 4 REORDER / 1 DIVERGE, drift ≥ 0.935. The single DIVERGE (Q10, pl template query, teacher-top1 at rank 14) is identical across vocab4e/12k/10k/8k → caused by the int4 *body*, not the vocab shrink; Q3 (pl rendering of an es instruction) actually *improves* rank 3→0 thanks to tokenizer decomposition replacing unks.

**Per-language models?** Rejected. The body is the cost (≈11.5 MB int4); a per-language model still carries it, so 5 models ≈ 5×(11.5 + ~1 MB shard) ≈ 60+ MB to keep all languages resident — worse than one shared 14.5 MB model — and it kills the shared space (cross-lingual retrieval, top1-xl 0.46-0.51 here, becomes impossible). The same goal (load less per language) is served by sharding the *embedding table* of the shared model, which at int4 costs ~0.2 MB/1k tokens — not worth the plumbing at these sizes.

**Updated deployment recommendation (supersedes §8 pick):** ship **vocab4s10** (`models/edge/nav_e5s_distill_vocab4s10.onnx` + `models/edge/nav_tok10/`) — **14.49 MB total, sp 0.895, 9/10 fresh queries in top-5**, max seq 320, no LUT. If 16 MB is acceptable, vocab4e buys back sp 0.901 / xl 0.512; if maximum robustness is wanted, vocab8 remains the conservative pick at 43.7 MB true footprint.

## Files

| File | Purpose |
|---|---|
| `openrouter_determinism.py` / `data/determinism_results.json` | §1 tests + results |
| `probe_drift.py` | backend-flip isolation probe |
| `embed_cache.py` / `data/embeddings_cache.db`, `data/cache_benchmark.json` | reusable cache + §2 benchmark |
| `profile_local.py` / `data/profile_results.json`, `data/bench_texts.json`, `data/groundtruth_fp32.npy` | §3 profiling + shared bench assets |
| `quantize_compare.py` / `models/model_int8.onnx(.data)`, `data/quantize_results.json` | §4 quantization + comparison |
| `edge_compare.py` / `data/edge_compare_results.json`, `data/edge_bench.jsonl`, `data/edge_gt_fp32.npy`, `models/edge/*_int8.onnx` | §6 edge-model comparison |
| `BASE_MODEL_RESEARCH.md` | base-model candidates + training recipe for TODO item 3 |
| `train_distill.py` / `data/distill_results.json`, `data/teacher_full_bge_m3.npy`, `data/train_texts.jsonl` | §7 distillation training + results |
| `models/nav-e5s-distill/` / `models/edge/nav_e5s_distill(.int8).onnx` | §7 trained student model (ST format) + ONNX exports |
| `quant_ladder.py` / `data/quant_ladder_results.json` | §8 size ladder (int4 / vocab pruning / static) |
| `models/edge/nav_e5s_distill_vocab8.onnx` (+ `_vocab4`, `_int4`, `_vocab_fp32`) / `nav_vocab_kept_ids.json` / `models/edge/nav_static/` | §8 ladder artifacts (vocab8 = §8 pick) |
| `models/edge/nav_e5s_distill_vocab4e.onnx` + `nav_e5s_distill_vocab4s{12,10,8}.onnx` / `models/edge/nav_tok{15,12,10,8}/` / `nav_vocab_kept_ids_{12,10,8}k.json` / `data/nav_token_freq.json` | §8.1 <15 MB rungs (vocab4s10 = current pick) |
| `data/fresh_audit_results.json` / `data/fresh_audit_teacher_q.npy` | §8.1 fresh-query audit (10 hand-written queries, teacher cached) |
| `discrepancy_demo.py` | live-query demo: fresh queries → top-5 retrieval, teacher (bge-m3) vs deployed student |
