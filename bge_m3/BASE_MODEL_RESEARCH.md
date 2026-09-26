# Base-model research — training own embeddings for navigation instructions

Date: 2026-09-26. Facts below verified via HF model cards / API on this date.
Goal: pick the base model to train a **domain embedding model for car-navigation instructions** — multilingual (en/de/fr/pl/es), edge-deployable (CPU-only SoC in the car, small RAM/flash), text similarity + cross-lingual matching + dedup of short instructions (30–732 chars, corpus `instructions_v3.jsonl`, 22k).

## Requirements

1. Native vocabulary/coverage for en, de, fr, **pl** (pl is the discriminator — many multilingual models degrade there; our langs are all high-resource European).
2. ≤ ~300M params, encoder-friendly for CPU int8 ONNX (transformer encoder, not decoder).
3. Permissive license (automotive/commercial: Apache-2.0 / MIT).
4. Good initialization for sentence-embedding training (contrastive/distillation) — raw MLM checkpoints need more data; embedding-pretrained checkpoints need less.

## Candidates — raw pretrained encoders (train embedding from scratch)

| Base | Params | License | Langs | Notes for our use |
|---|---|---|---|---|
| **EuroBERT/EuroBERT-210m** | 310M (210M non-emb) | Apache-2.0 | 15 incl. **en/de/fr/pl/es** | Modern RoBERTa-class (RoPE, RMSNorm, 8k ctx), European-language focus, all 5 of our langs in pretraining set. Best raw-base headroom. |
| FacebookAI/xlm-roberta-base | 278M | MIT | 100 | The proven standard — lineage of LaBSE / multilingual-e5 / BGE-m3 itself. Every ST training recipe works out of the box; pl coverage good. |
| microsoft/mdeberta-v3-base | 279M | MIT | 101 | Best classification scores (XNLI) but ELECTRA-style pretraining + disentangled attention → known **weak raw sentence-similarity** representations and CPU/quantization-unfriendly custom ops. Not recommended for ST. |
| ModernBERT-base | 149M | Apache-2.0 | en only | Fastest encoder on CPU but **monolingual** → out. |

## Candidates — embedding-pretrained inits (continue fine-tune on domain; fastest path)

| Init | Params | Dims | License | Notes |
|---|---|---|---|---|
| **intfloat/multilingual-e5-small** | 118M | 384 | MIT | XLM-R lineage, already embedding-tuned, ships ONNX; our edge comparison (RESULTS.md §6) shows quality/latency. Continue-training needs the least data. |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 118M | 384 | Apache-2.0 | Same class; slightly weaker on retrieval than e5-small typically. |
| Qwen/Qwen3-Embedding-0.6B | 596M | 1024 (MRL 32–1024) | Apache-2.0 | Decoder-based embedder; too heavy as the deployed student but excellent **teacher** (or MRL template to imitate). |

## Teachers for distillation (not deployed)

- **BAAI/bge-m3** (local GPU fp16, proven deterministic for our use — RESULTS.md §3): 1024-d, strong cross-lingual. Teacher scores already reproducible via `data/` pipeline.
- Qwen3-Embedding-0.6B: higher MTEB-multilingual, MRL-native — good second opinion / ensemble teacher.

## Recommendation

1. **Fast path (recommended first experiment): fine-tune `multilingual-e5-small`** on our corpus by distilling BGE-M3 (or Qwen3) cosine structure + domain contrastive pairs. 118M/384d is comfortably edge-sized; least data-hungry; direct drop-in for the ONNX int8 pipeline already benchmarked.
2. **Max-quality path: train from `EuroBERT-210m`** — best language fit (all 5 langs first-class, European focus) and modern architecture; needs more training data/steps than the e5-small path; export to ONNX int8 once trained.
3. XLM-R-base = conservative fallback if EuroBERT training proves unstable; mDeBERTa excluded (ST weakness + edge-unfriendly ops).

## Training recipe sketch (applies to either path)

- **Objective:** InfoNCE with teacher-scored soft labels (KL over teacher cosine-sim matrix — "distillation with hard negatives") + optional **Matryoshka loss** over dims {64, 128, 256, full} so the deployed dim can shrink on-device.
- **Positives:** (a) cross-lingual same-intent pairs (same origin/destination/constraints, different language — derivable from corpus metadata), (b) LLM paraphrases (corpus is already LLM-harvested; same harvesters can generate positives), (c) instruction↔(origin,destination) pairs.
- **Negatives:** same-language different-route instructions (hard), random cross-language (easy).
- **Data scale:** 22k instructions → ~100k+ pairs feasible; distillation lets every text pair carry signal without labels.
- **Eval:** the §6 harness (Spearman/Jaccard@10/MRR vs BGE-M3 GT + cross-lingual top-1 agreement) re-used verbatim as the training metric — no new infra needed.
- **Deploy:** ONNX dynamic int8 (same as Phase 3/§6 pipeline); 384d × fp16/int8 vector storage.

## Open items

- Verify EuroBERT-210m fine-tunes stably with ST/contrastive training (no public sentence-embedding fine-tune of it yet in our check).
- Decide teacher: bge-m3 (proven, matches current GT) vs Qwen3-0.6B (higher ceiling) — can distill from both and average.
