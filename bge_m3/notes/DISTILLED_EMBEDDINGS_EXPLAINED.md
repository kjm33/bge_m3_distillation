# Notes: distilled navigation embeddings — plain-language summary

Companion to `RESULTS.md` §7 (authoritative numbers) and `BASE_MODEL_RESEARCH.md`. Written 2026-09-26 after the distillation run (config `tau0.03_bs64_ep8`, best epoch 6).

## 1. What we did, in plain language

**The problem.** BGE-M3 is our big high-quality embedding model (2.3 GB, server GPU), but a car's navigation system has a small CPU — it needs something tiny and fast. The stock-model comparison (§6) showed every small off-the-shelf model disagrees heavily with BGE-M3 — like a different librarian who organizes books differently. If the device picks "nearest instruction X" while the server model would pick "Y", behavior becomes inconsistent.

**The idea: distillation (teacher–student).** Instead of searching for a small model that *happens* to think like BGE-M3, we trained one to think like it:

1. **Teacher** (BGE-M3) read all 21,447 instructions and produced its similarity structure — which instructions are similar to which, across all 5 languages.
2. **Student** (multilingual-e5-small, 118M params, 384-dim) was trained for 25 minutes on one RTX 3090 with one goal: *arrange instructions the same way the teacher does*.
3. 512 instructions were held out from training — the scores below measure real understanding, not memorization.

**What the metrics mean** (before → after training):

| Metric | Question it answers | stock e5-small | trained |
|---|---|---|---|
| Spearman 0.57 → 0.91 | Ranks all instruction pairs the same way as the teacher? | mediocre | near-identical |
| Jaccard@10 0.36 → 0.62 | "10 most similar" lists overlap with the teacher's? | only 1/3 | ~2/3 |
| Top-1 agreement 0.42 → 0.68 | Picks the same #1 match? | often wrong | usually right |
| **Cross-lingual top-1: 0.01 → 0.55** | "Turn right after the bridge" (EN) matches its FR/PL/DE/ES equivalent as the teacher would? | ~zero — separate language worlds | more than half — languages unified |

The cross-lingual jump is the headline: we gave it **zero translation pairs** — it absorbed language alignment purely by imitating the teacher's similarity structure.

**Deployment win.** Training rewires the model, it doesn't grow it: same size/speed as stock e5-small (112.6 MB, ~4.7 ms per instruction on CPU), but **74% more faithful to BGE-M3**. Robust to int8 compression (0.91 → 0.90), unlike LaBSE which breaks (cos 0.52). Matryoshka-trained: vectors can be cut from 384 to 128–256 dims later for half the memory, without retraining.

**Bottom line:** for 25 minutes of training on one GPU we own a 113 MB model that behaves like a 2.3 GB model on our navigation domain. The "small XOR faithful" dilemma is gone. Artifacts: `models/nav-e5s-distill/` (+ int8 ONNX in `models/edge/`); retrain cheaply whenever the corpus grows.

## 2. Why e5-small as the student

1. **Proven edge behavior** — best quality-per-byte in our §6 benchmark among true edge candidates: highest GT agreement of all sub-300M models (sp 0.572, ahead of minilm 0.471 and qwen3-0.6B 0.420); MIT license; 118M/384d.
2. **Embedding-pretrained init** — already knows "sentence similarity" from contrastive pretraining, so distillation needs little data/steps. A raw MLM checkpoint (EuroBERT-210m, the max-quality alternative) would need more of both — it remains the follow-up path if >0.95 agreement is ever needed.
3. **Multilingual with good Polish** — XLM-R lineage covers all 5 languages. LaBSE covers them too but is 471M/1.75 GB and breaks under int8 — disqualified for our target.
4. **Drop-in compatibility** — same architecture/pipeline as the already-benchmarked stock model: apples-to-apples before/after comparison, and it deploys in the exact same int8 ONNX wrapper.

## 3. BGE-M3 (teacher) vs distilled e5-small int8 (deployed student)

BGE-M3 is the reference, so "agreement" measures how faithfully the student copies it.

| | BGE-M3 int8 (CPU) | nav-e5s-distill int8 | ratio |
|---|---|---|---|
| Model size | 542 MB | **112.6 MB** | 4.8× smaller |
| Load time | 0.40 s | **0.28 s** | 1.4× faster |
| Single-text latency (p50) | 41.0 ms | **4.7 ms** | **8.8× faster** |
| Throughput (bs8, ~23 tok/text) | ~30 texts/s | **212 texts/s** | ~7× faster |
| RAM delta | ~1.3 GB | **~94 MB** | ~14× less |
| Vector dim / storage | 1024d / 4 KB | **384d / 1.5 KB** (MRL → 128d / 512 B) | 2.7× smaller index |

Student fidelity (512 held-out texts): Spearman **0.896**, Jaccard@10 **0.599**, MRR@10 0.785, top-1 0.664, cross-lingual top-1 **0.50**, ARI 0.370.

**Reading:** for ~1/5 the size and ~1/9 the latency, the student reproduces ~90% of the teacher's ranking structure and half of its exact top-1 picks; cross-lingual behavior — BGE-M3's key strength — is substantially preserved (0.50 vs stock e5-small's 0.01). The remaining gap is the price of 20× compression. If exact fidelity is ever needed, keep BGE-M3 (GPU fp16 server-side, or its 542 MB int8).

## 4. Full comparison — distilled model vs everything we checked

Quality vs BGE-M3 GT (same 512-text held-out bench, §6 + §7):

| model | params | sp (20k) | Jac@10 | MRR@10 | top-1 | top-1 cross-ling. | ARI-50 |
|---|---|---|---|---|---|---|---|
| **nav-e5s-distill (ours)** | **118M** | **0.908** | **0.623** | **0.796** | **0.678** | **0.548** | **0.369** |
| labse | 471M | 0.725 | 0.350 | 0.529 | 0.383 | 0.309 | 0.222 |
| e5_base | 278M | 0.571 | 0.388 | 0.598 | 0.478 | 0.059 | 0.255 |
| e5_small (stock) | 118M | 0.572 | 0.356 | 0.551 | 0.424 | 0.012 | 0.252 |
| potion (static) | – | 0.501 | 0.303 | 0.472 | 0.350 | 0.036 | 0.200 |
| minilm | 118M | 0.471 | 0.295 | 0.446 | 0.316 | 0.202 | 0.218 |
| qwen3-0.6B | 596M | 0.420 | 0.365 | 0.543 | 0.422 | 0.095 | 0.266 |

The distilled model wins **every column** with the same 118M params as stock e5-small/minilm, beating models 4–5× its size (LaBSE, Qwen3-0.6B).

Edge deployment package (CPU, int8 ONNX where available):

| model | int8 size | bs8 texts/s | p50 ms | Jac@10 vs GT (int8) |
|---|---|---|---|---|
| **nav-e5s-distill int8** | **112.6 MB** | 212 | **4.7** | **0.599** |
| e5_small int8 | 112.6 MB | 230 | 5.1 | 0.345 |
| minilm int8 | 112.7 MB | 241 | 3.9 | 0.294 |
| e5_base int8 | 265.3 MB | 83 | 12.8 | 0.360 |
| labse int8 ⚠ | 449.4 MB | 66 | 51.2 | 0.316 (vectors broken: cos 0.52) |
| potion (static) | ~512 MB | 29,211 | 0.2 | 0.303* |
| bge_m3 int8 | 542 MB | ~30 | 41.0 | (is the GT itself) |
| qwen3-0.6B | 596M+ | 3.7 (fp32) | 144.4 | 0.365* |

\* fp32 values (no int8 export tested).

**Takeaways**

- nav-e5s-distill int8 costs **exactly the same** as stock e5_small int8 (same architecture → same 112.6 MB, same speed class) but delivers **74% higher neighbor agreement**.
- Only potion is faster (sub-ms static lookup) but it's prefilter-tier — weakest structure agreement, different use case.
- Against bge-m3 int8 (the "just ship the big model" option): 4.8× smaller, ~9× lower latency, ~14× less RAM, at ~90% rank fidelity.
- Next quality levers if ever needed: EuroBERT-210m student, Qwen3 ensemble teacher (§7.3 / BASE_MODEL_RESEARCH.md).

## 5. Update: we shrank it 4× more — for free (§8)

The 112.6 MB above is not the end. Inspection showed the model is a "dictionary with a small brain attached": **73% of its 131M parameters are the multilingual dictionary (embedding table), and our navigation instructions only ever use 15,199 of its 250,002 words**. So we deleted the unused 94% of the dictionary and rebuilt the model around the survivors.

| version | size | speed (p50) | quality (sp / jac@10) | note |
|---|---|---|---|---|
| int8 (§7) | 112.6 MB | 4.7 ms | 0.896 / 0.599 | previous pick |
| **+ dictionary pruned (vocab8)** | **26.6 MB** | 4.3 ms | **0.896 / 0.599 — identical** | **new deployment pick** |
| pruned + 4-bit layers (vocab4) | 17.8 MB | 5.2 ms | 0.901 / 0.611 | if every MB counts |
| static lookup table | 14.9 MB | 0.15 ms | 0.468 / 0.247 | prefilter only |

In plain terms: the pruned model **answers exactly the same as before** on anything written with the vocabulary our domain uses (verified: same neighbor lists, same rankings, cosine 1.0 against the unpruned model on the held-out test set). The trade-off we accepted: a genuinely new word outside the kept vocabulary falls back to "unknown" — irrelevant for the closed world of navigation instructions (user-confirmed), but this model should not be repurposed for open-domain text.

Why it's safe here and not universally: the kept rows are byte-identical to the original ones — pruning removes words, it doesn't retrain anything. Quality can only change through words we never see at runtime.

Final standing vs the original BGE-M3 teacher: **26.6 MB (20× smaller), 4.3 ms (9.5× faster), ~53 MB RAM (25× less), at ~90% of its ranking fidelity** — a 2.3 GB server model's behavior, pocket-sized.
