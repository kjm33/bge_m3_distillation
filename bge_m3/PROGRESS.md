# PROGRESS — bge_m3 edge embeddings for navigation instructions

Goal: a small, fast CPU embedding model faithful to BGE-M3 for car-navigation
instructions in 5 languages (en/de/fr/pl/es), corpus `instructions_v3.jsonl`
(21,959 texts), 512 held-out bench (`data/edge_bench.jsonl`), GT = BGE-M3 fp32
CUDA CLS (`data/edge_gt_fp32.npy`). Full details: RESULTS.md (§ refs below).

## Status: DONE — deployment pick

| tier | artifacts | total size | sp 20k | jac@10 | top1-xl | p50 |
|---|---|---|---|---|---|---|
| conservative | `nav_e5s_distill_vocab8.onnx` + full tokenizer + LUT | 43.7 MB | 0.896 | 0.599 | 0.500 | 4.3 ms |
| max quality ≤16 MB | `nav_e5s_distill_vocab4e.onnx` + `nav_tok15/` | 15.65 MB | 0.901 | 0.607 | 0.512 | 5.3 ms |
| **current pick (<15 MB)** | `nav_e5s_distill_vocab4s10.onnx` + `nav_tok10/` | **14.49 MB** | 0.895 | 0.603 | 0.464 | 5.1 ms |

Reference: student fp32/ONNX int8 sp 0.908/0.896, BGE-M3 int8 ONNX 542 MB p50 41 ms.

Headline: the distilled model delivers **~90% of BGE-M3's ranking fidelity at
1/37th the size and ~9× lower latency** (14.49 MB / 5.1 ms vs 542 MB / 41 ms) —
better deployment economics, not better embeddings; BGE-M3 remains the quality
reference (all student metrics measure agreement with it).

How "~90% ranking fidelity" is measured:
- **§7/§8 harness** (`quant_ladder.py` `quality_report`/`make_pairs`, seed 42):
  512 held-out bench texts embedded by teacher (BGE-M3 fp32 CUDA,
  `data/edge_gt_fp32.npy`) and student; **20k random pairs**; Spearman
  correlation between teacher and student cosine vectors. vocab4s10 = 0.8945,
  vocab4e/int4 = 0.901, fp32 student = 0.908 → "~90%".
- **Cross-similarity audit** (`data/cross_sim_bge_m3_vs_distill.json`):
  130,816 pairs → Spearman 0.908 / Pearson 0.919 (mono 0.950, xl 0.896).
- Caveat: Spearman over pair scores is a *proxy* for ranking agreement. Direct
  ranking overlap is stricter: jaccard@10 vs teacher top-10 ≈ 0.60, MRR@10 of
  teacher top-1 ≈ 0.78, top-1 agreement ≈ 0.66 (0.50 xl), per-query top-10
  overlap median 0.667. Disagreements are reorders among plausible neighbors;
  zero hard contradictions (no pair teacher ≥ 0.70 & student < 0.60).

## Chronology

1. **§1–3 Infrastructure** — OpenRouter bge-m3 API is per-backend deterministic
   but not bit-exact; local cache mandatory (1343× faster); local fp16 1378 t/s.
2. **§4 BGE-M3 int8 ONNX** — 542 MB, p50 41 ms, jac@10 0.778 (impractical for edge).
3. **§6 stock-model screen** — labse best sp 0.725, jac ≤ 0.39, xl ≤ 0.31 → no
   drop-in replacement; distillation needed.
4. **§7 distillation** (`train_distill.py`) — teacher BGE-M3 → student
   intfloat/multilingual-e5-small (118M/384d): sym-KL τ=0.03 on cosine matrices,
   Matryoshka {64,128,256,384}, hard-neighbor batches, 8 ep, 25 min RTX 3090.
   Result: sp 0.908, jac@10 0.623, MRR@10 0.796, top1 0.678, top1-xl 0.548,
   ARI 0.369; int8 ONNX 112.6 MB p50 4.7 ms sp 0.897.
5. **Cross-similarity audit** (`data/cross_sim_bge_m3_vs_distill.json`) —
   130,816 pairs: Spearman 0.908, mono 0.950 vs xl 0.896, zero hard
   disagreements, calibration ≈ identity (+0.02–0.03) → port thresholds +0.03,
   keep hard score floor ~0.6.
6. **Discrepancy demo** (`discrepancy_demo.py`, 10 hand-written fresh queries) —
   6 EXACT / 3 REORDER / 1 IN_TOP10 / 0 DIVERGE vs live teacher; 4.3 ms/query.
7. **§8 size ladder** (`quant_ladder.py`) — vocab pruning 250,002→15,276
   (corpus-exact ∪ specials ∪ chars): vocab8 26.6 MB **zero quality loss**;
   vocab4 17.8 MB sp 0.901; static model2vec 14.9 MB sp 0.465 (prefilter-tier
   only).
8. **§8.1 <15 MB experiment** (2026-09-26) — true footprint = onnx + tokenizer
   (full `tokenizer.json` is 17 MB; vocab8 really ships 43.7 MB). Two new levers:
   - **pruned tokenizer** (`nav_tok*`, 0.54–0.75 MB): emits new ids natively
     (no LUT), byte-identical ids to full+LUT on 21,959 corpus + 512 bench
     (0/0 mismatches); fresh words decompose instead of `<unk>` (strictly
     better: unk 3→0 on demo, Q3 rank 3→0).
   - **int4 embedding tables**: exact per-row symmetric int4 (uint8 nibble
     pairs + per-row fp32 scales, in-graph opset-17 decode); pos table trimmed
     514→320 (max_len 256); MatMulNBits int4 body ≈ 11.5 MB fixed.
   - Rungs: vocab4e 15.65 MB sp 0.901; vocab4s12 14.95 MB sp 0.8954;
     **vocab4s10 14.49 MB sp 0.8945** (floor 0.89 ✓, budget ✓); vocab4s8
     14.04 MB sp 0.8882 ✗ floor.
   - Fresh-query audit (`--stage fresh_audit`, teacher live fp32 CUDA, cached):
     vocab8 5 EXACT/5 REORDER/0 DIVERGE; all int4 rungs 5/4/0/1 — the single
     DIVERGE (Q10 pl-template, rank 14) identical across all int4 variants
     incl. full-vocab vocab4e → int4-body effect, not vocab shrink.

## Decisions & rejected alternatives

- **Per-language models: rejected** — body (~11.5 MB int4) dominates; 5 models
  ≈ 60+ MB resident and kills the shared space (cross-lingual retrieval).
  Same goal served (if ever needed) by sharding the shared model's embedding
  table (~0.2 MB/1k tokens at int4).
- **Generalist stage (Shitao/bge-m3-data, 24 GB): skipped** — model is used
  only for navigation instructions; nav-domain fidelity is the objective.
  Note: that data has no Polish anyway.
- **Static (model2vec) tier: prefilter/dedup only** — sp 0.465, xl 0.0.
- **Layer-drop 12→10 + re-distill: not needed** — 10k rung met budget+floor.
- Cross-lingual = nice-to-have, not hard (but preserved by all picks).

## Open items / ideas

- **Quality ceiling push**: EuroBERT-210m student (all 5 langs first-class)
  projected >0.95 sp; recipe unchanged (`train_distill.py`).
- QAT if int4 becomes the long-term path (post-training int4 already only
  costs ~0.005 sp; the Q10 rank-4→14 wobble is the visible symptom).
- Guardrails to port from analysis: dedup threshold +0.03 on student scores;
  hard no-match floor ~0.6.
- MRL 128d corpus vectors halve index storage (1.5 KB→512 B/text).

## Conventions (critical for reuse)

- e5 student: `"query: "` prefix on ALL texts; mean pool; always l2norm.
- BGE-M3 teacher: no prefix, CLS pooling.
- ONNX student: feed `token_type_ids` zeros for short single texts.
- vocab4s10 deployment contract: pruned tokenizer with `no_padding()` +
  truncation to 256 (pos table hard-limits at 320).
