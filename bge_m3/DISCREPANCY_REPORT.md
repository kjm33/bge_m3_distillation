# Discrepancy report — BGE-M3 vs distilled e5-small "in practice"

Date: 2026-09-26 · Companion to `RESULTS.md` §7 (aggregate metrics) and `data/cross_sim_bge_m3_vs_distill.json` (130k-pair cross-similarity).
Question answered here: **when a real query hits both models, where do they disagree, and what does that do to retrieval and dedup?**

## Setup

- **Corpus (search space):** 512 held-out bench instructions, 5 languages, embedded with both models (`data/edge_gt_fp32.npy`, `data/nav_e5s_distill_torch_emb.npy`).
- **Queries:** 10 fresh hand-written instructions — 1 verbatim control, paraphrases, cross-lingual renderings, constraint-theme query, novel intent, 4 languages. None were fed to either model before.
- **Teacher = BGE-M3** fp32 CUDA, CLS (live, `transformers`). **Student = `nav-e5s-distill` int8 ONNX CPU**, bs=1, `"query: "` prefix (live, the actual deployment artifact from `RESULTS.md` §7.2).
- Reproduce with `discrepancy_demo.py` → `data/discrepancy_demo_results.json`.

## Headline

| verdict (student vs teacher) | count | meaning |
|---|---|---|
| EXACT — same #1 neighbor | 6/10 | behavior identical where it matters most |
| REORDER — teacher's #1 inside student's top-5 | 3/10 | same candidate set, different order |
| IN_TOP10 | 1/10 | teacher's #1 still on page 1 for the student |
| DIVERGE — teacher's #1 lost | **0/10** | no query lost its answer |

Query latency: teacher 25 ms/query (GPU fp32, batched) vs student **4.3 ms/query (CPU int8, bs=1)**.

## Case walkthrough

### Q1 control — verbatim corpus text → EXACT
Both models return the exact original at cos 1.000 / 0.980. Sanity passed: exact duplicates survive the distilled+int8 pipeline.

### Q2 paraphrase (en) → EXACT, overlap 5/5
"I need to get from Vienna to Budapest, take a break every 100 km and every two hours" → both pick the corpus original ("go from Vienna to Budapest with breaks after each 100 km…") as #1 (0.943 / 0.958). **Paraphrase matching is intact.**

### Q3 / Q4 cross-lingual renderings → EXACT
- Q4 (German rendering of an en text): both return the English original as #1 (0.851 / 0.823). **Cross-lingual intent matching works in both.**
- Q3 (Polish rendering of an es text) is the most interesting case: both models' top-2 are **identical** (toll-avoidance + break instructions), but neither ranks the Spanish original #1 — teacher: rank 6 (0.654), student: rank 16 (0.671). Both retrieve the *theme* correctly; the exact translation-pair is mid-ranked in both. Student demotes it further — the residual cross-lingual gap (top1-xl 0.548, not 1.0) shows up exactly here.

### Q7 (de, Augsburg→Regensburg, charging every 200 km) — the one real ranking split
- Teacher #1: `navigiere von Shenzhen nach Shuizhai mit Lade stopps alle 300 km…` — a **semantic** match (same intent: charging stops at intervals), no shared cities.
- Student #1: `Navigiere mich von München nach Hamburg … Landstraßen … Ladepause` — a **template** match (same phrasing pattern "Navigiere … Ladepause"), different intent details.
- Teacher's #1 is still rank ~6–8 for the student (IN_TOP10). Teacher's #5 (`Köln→München mit Zwischenstopp in Regensburg` — the only text naming Regensburg) is student's #4.
- **Pattern:** the student leans on lexical/template similarity a bit more than the teacher; the teacher weighs intent (constraint type) slightly more. Both lists are defensible; they differ in *emphasis*.

### Q8 (fr, Lyon→Marseille, avoid tolls + coffee near Valence) — REORDER, 4/5 overlap
Teacher's top-5 is Lyon-anchored; student promotes `Paris→Lyon … par Dijon pour prendre un café` — the only candidate matching **both** toll-avoidance and a coffee stop. Here the student's reordering arguably matches the *constraints* better while the teacher matches the *cities* better. Same pool, different priority.

### Q10 (Gdańsk→Sopot, no corpus match) — the stress case, overlap 2/5
Neither model has a true neighbor (all cosines < 0.68). Teacher retreats to an English bare template (`route from Warsaw to Krakow`), student to Polish texts — its #1 actually names Gdańsk. **When nothing matches, the two models flounder differently**; if the product shows "best match" anyway, users will see different filler. Prefer a hard score floor (e.g. cos < 0.6 → "no match") — then this case becomes identical non-matches.

## Dedup at a fixed threshold (cos ≥ 0.75) — the most practical discrepancy

Across all 10 queries:

- Teacher flags **0 pairs that the student misses** (T-only extras: none, anywhere).
- Student flags extra pairs in 5/10 queries — all thematically adjacent (other EV-charging / toll-avoidance / break instructions at 0.75–0.82).
- Cause: student cosines run **~+0.03 higher** at the top end (matches the 130k-pair calibration: teacher bin [0.70,0.80) → student mean 0.763).

**Practical rule:** port dedup thresholds with a **+0.03 shift** (e.g. teacher 0.75 → student 0.78) and the flag sets converge; the safe direction is already safe — nothing the teacher calls a duplicate is missed by the student (dup-recall preserved; only dup-precision is slightly looser).

## Systematic effects observed (consistent with the 130k-pair analysis)

1. **Score inflation, not noise:** student ≈ teacher cosine + 0.02–0.09, monotone (Pearson 0.919). Shift thresholds, don't re-tune from scratch.
2. **Rank stability at the top, shuffle below:** #1 agreed 6/10 here (small sample; corpus-wide top-1 agreement 0.678), teacher's #1 present in student top-10 in 10/10.
3. **Student is mildly template-biased, teacher mildly intent-biased** (Q7, Q9, Q10) — discrepancies appear as *reordering among plausible matches*, not as absurd neighbors.
4. **Cross-lingual is good but not teacher-grade:** exact translation pairs can sink to rank ~16 for the student where the teacher keeps them ~6.
5. **Both models agree on "no match" cases** only if you impose a score floor; without one, their fallback picks diverge the most.

## Bottom line

Running identical live queries through both models: the student (113 MB, 4.3 ms, CPU int8) reproduces the teacher's (2.1 GB, GPU) #1 answer in 6/10 cases and keeps it in the top-10 in 10/10; disagreements are reorderings among semantically plausible candidates plus a mild cosine inflation. For retrieval-with-floor, dedup-with-shifted-threshold, and clustering, the deployed student is a practical stand-in; the residual risk concentrates in cross-lingual exact-pair ranking and in "nothing matches" fallback behavior.

## Files

| File | Purpose |
|---|---|
| `discrepancy_demo.py` | live demo: fresh queries → both models → JSON |
| `data/discrepancy_demo_results.json` | per-query top-5s, verdicts, dedup sets, latencies |
| `data/cross_sim_bge_m3_vs_distill.json` | 130,816-pair cross-similarity, calibration, threshold transfer |
