#!/usr/bin/env python3
"""Practical discrepancy demo: fresh queries -> BGE-M3 (teacher) vs nav-e5s-distill int8 ONNX (deployed).

For each hand-written query (NOT verbatim corpus texts):
  - embed live with BGE-M3 fp32 CUDA (CLS, no prefix)  [train_distill.py methodology]
  - embed live with nav-e5s-distill int8 ONNX CPU (mean pool, "query: " prefix)  [deployment artifact]
  - top-5 retrieval from the 512-instruction bench corpus under each model
  - verdict: EXACT (same #1) / REORDER (teacher #1 inside student top-5) / DIVERGE (outside)
  - near-duplicate flags at cosine >= 0.75 (dedup use case), symmetric difference
  - per-query latency of both models

Outputs: data/discrepancy_demo_results.json  (DISCREPANCY_REPORT.md is written by hand from this)
"""
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GT_PATH = DATA / "edge_gt_fp32.npy"                      # BGE-M3 fp32, 512 bench
STUDENT_EMB_PATH = DATA / "nav_e5s_distill_torch_emb.npy"  # student torch fp32, 512 bench
ONNX_INT8 = HERE / "models" / "edge" / "nav_e5s_distill_int8.onnx"
STUDENT_DIR = HERE / "models" / "nav-e5s-distill"
TEACHER_REPO = "BAAI/bge-m3"
E5_PREFIX = "query: "
TOPK = 5
DEDUP_T = 0.75

QUERIES = [
    dict(id="Q1_control_verbatim", kind="control-exact-duplicate (en)",
         text="route from San Antonio to Cedar Park avoiding tolls"),
    dict(id="Q2_paraphrase_en", kind="paraphrase of en corpus text (en)",
         text="I need to get from Vienna to Budapest, take a break every 100 km and every two hours"),
    dict(id="Q3_xling_pl_of_es", kind="Polish rendering of es corpus text (pl)",
         text="Poprowadź mnie z Barcelony do Saragossy bez płatnych dróg, z przerwą na obiad w Lleidzie"),
    dict(id="Q4_xling_de_of_en", kind="German rendering of en corpus text (de)",
         text="Führ mich von San Antonio nach Cedar Park und meide alle Mautstraßen"),
    dict(id="Q5_theme_ev", kind="constraint-theme only, no city anchor (en)",
         text="I'm driving an EV, plan charging stops every 300 km and a coffee break on the way"),
    dict(id="Q6_offcorpus_intent", kind="novel intent, no near neighbor expected (en)",
         text="find me the nearest parking garage with a car wash in Zagreb"),
    dict(id="Q7_de_ev", kind="new city pair, EV constraint (de)",
         text="Navigiere von Augsburg nach Regensburg und plane alle 200 km eine Ladepause"),
    dict(id="Q8_fr_tolls", kind="new city pair, tolls + coffee (fr)",
         text="Trajet de Lyon à Marseille en évitant les péages, avec un arrêt café vers Valence"),
    dict(id="Q9_es_fastest", kind="new city pair, fastest route (es)",
         text="llévame de Valencia a Alicante por la ruta más rápida"),
    dict(id="Q10_template_pl", kind="corpus template style, pl cities, en wording (en)",
         text="route from Gdańsk to Sopot via the fastest route"),
]


def l2norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_corpus():
    texts, langs = [], []
    with open(DATA / "edge_bench.jsonl") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            langs.append(d["language"])
    return texts, np.array(langs)


def embed_teacher(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER_REPO)
    model = AutoModel.from_pretrained(TEACHER_REPO).to("cuda").eval()
    outs, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            enc = tok(texts[i:i + 8], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to("cuda")
            outs.append(model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
    wall = time.perf_counter() - t0
    return l2norm(np.concatenate(outs)), wall


def embed_student_int8(texts):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(STUDENT_DIR))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(ONNX_INT8), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    ts = [E5_PREFIX + t for t in texts]
    outs, t0 = [], time.perf_counter()
    for t in ts:  # bs=1: per-query latency as deployed
        enc = tok([t], padding=True, truncation=True, max_length=256, return_tensors="np")
        feed = {n: enc[n].astype(np.int64) for n in names if n in enc}
        if "token_type_ids" in names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        h = next(o for o in sess.run(None, feed) if o.ndim == 3)
        mask = enc["attention_mask"].astype(np.float32)
        m_ = mask[:, :, None]
        v = (h * m_).sum(1) / np.maximum(m_.sum(1), 1.0)
        outs.append(v.astype(np.float32))
    wall = time.perf_counter() - t0
    return l2norm(np.concatenate(outs)), wall


def top_list(q, C, texts, langs, k=TOPK):
    sims = C @ q
    order = np.argsort(-sims)[:k]
    return [dict(idx=int(j), lang=str(langs[j]), cos=round(float(sims[j]), 4),
                 text=texts[j][:110]) for j in order]


def main():
    texts, langs = load_corpus()
    gt = l2norm(np.load(GT_PATH))            # teacher corpus embeddings (fp32 CUDA, exact)
    st = l2norm(np.load(STUDENT_EMB_PATH))   # student corpus embeddings (torch fp32)
    qtexts = [q["text"] for q in QUERIES]

    tq, w_teacher = embed_teacher(qtexts)
    sq, w_student = embed_student_int8(qtexts)

    results = []
    for meta, qv_t, qv_s in zip(QUERIES, tq, sq):
        top_t = top_list(qv_t, gt, texts, langs)
        top_s = top_list(qv_s, st, texts, langs)
        t_top1 = top_t[0]["idx"]
        s_rank_of_t_top1 = next((i for i, d in enumerate(top_s) if d["idx"] == t_top1), None)
        if s_rank_of_t_top1 == 0:
            verdict = "EXACT"
        elif s_rank_of_t_top1 is not None:
            verdict = "REORDER"
        else:
            o10 = {d["idx"] for d in top_list(qv_s, st, texts, langs, k=10)}
            verdict = "IN_TOP10" if t_top1 in o10 else "DIVERGE"
        # dedup at fixed threshold
        dup_t = np.where(gt @ qv_t >= DEDUP_T)[0]
        dup_s = np.where(st @ qv_s >= DEDUP_T)[0]
        results.append(dict(
            id=meta["id"], kind=meta["kind"], text=meta["text"], verdict=verdict,
            overlap_at5=len({d['idx'] for d in top_t} & {d['idx'] for d in top_s}),
            teacher_top1_rank_in_student=s_rank_of_t_top1,
            teacher_top5=top_t, student_top5=top_s,
            dedup_teacher=[dict(idx=int(j), lang=str(langs[j]), cos=round(float(gt[j] @ qv_t), 3),
                                text=texts[j][:90]) for j in dup_t],
            dedup_student=[dict(idx=int(j), lang=str(langs[j]), cos=round(float(st[j] @ qv_s), 3),
                                text=texts[j][:90]) for j in dup_s],
        ))

    out = dict(
        meta=dict(n_corpus=len(texts), topk=TOPK, dedup_threshold=DEDUP_T,
                  teacher="BAAI/bge-m3 fp32 CUDA CLS (live)",
                  student="nav-e5s-distill int8 ONNX CPU (live, deployed artifact)",
                  corpus_emb=dict(teacher="data/edge_gt_fp32.npy (cached fp32)",
                                  student="data/nav_e5s_distill_torch_emb.npy (cached torch fp32)"),
                  wall_s=dict(teacher_all_queries=round(w_teacher, 3),
                              student_all_queries=round(w_student, 3))),
        results=results,
        summary=dict(verdicts={v: sum(1 for r in results if r["verdict"] == v)
                               for v in ("EXACT", "REORDER", "IN_TOP10", "DIVERGE")}),
    )
    with open(DATA / "discrepancy_demo_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["summary"], indent=2))
    for r in results:
        print(f"\n== {r['id']} [{r['verdict']}] overlap@5={r['overlap_at5']}")
        print(f"   Q: {r['text'][:95]}")
        print(f"   T#1 ({r['teacher_top5'][0]['cos']:.3f} [{r['teacher_top5'][0]['lang']}]) {r['teacher_top5'][0]['text'][:80]}")
        print(f"   S#1 ({r['student_top5'][0]['cos']:.3f} [{r['student_top5'][0]['lang']}]) {r['student_top5'][0]['text'][:80]}")


if __name__ == "__main__":
    main()
