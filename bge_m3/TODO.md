TODO:
- [x] compare other emebddings which could be used in a edge device - car navigation. Please include also Matryoshka embeddings. Maybe there are good enough models with much lower resource requirements (RESULTS.md §6)
- [x] research which base model is the best candidate to train own embeddings for navigation instructions (multilingual en/de/fr/pl/es, edge/CPU deployable; RoBERTa-class and smaller) (BASE_MODEL_RESEARCH.md)
- [x] train own embeddings on instructions_v3.jsonl (contrastive fine-tune / distill from BGE-M3 or Qwen3-Embedding) (RESULTS.md §7 — bge-m3 → e5-small distillation, sp 0.91 / cross-lingual top1 0.55, int8 112.6MB)
- [x] quantize the final model even more (RESULTS.md §8 — vocab pruning 250k→15.3k tokens: 26.6MB at identical quality; int4 17.8MB; static 14.9MB prefilter-tier)
