"""Quantization ladder benchmark for the distilled nav-e5s embedding model
(models/nav-e5s-distill, 384-d mean-pool, XLM-R 12L/384h student of bge-m3).

Rungs (runnable separately, results merged into data/quant_ladder_results.json):
  baseline - re-bench the existing models/edge/nav_e5s_distill_int8.onnx with THIS
             script's bench protocol; copies train_distill reference numbers alongside
  int4     - fp32 graph -> MatMulNBits (block 64, 4-bit) -> quantize_dynamic
             (embeddings/Gather become int8, MatMuls become 4-bit)
  vocab    - prune word embeddings to the ~16k tokens the corpus actually uses
             (full tokenizer + id remap at runtime), fp32 export, sanity gate
             (cos-vs-torch >= 0.9999), then quantize_dynamic -> int8
  vocab4   - pruned fp32 graph -> MatMulNBits -> quantize_dynamic -> int4+int8
  static   - model2vec-style static distillation (single-token forward passes,
             mean-pool) sliced to the kept vocab; pure embedding-lookup model

Run from bge_m3/:  .venv/bin/python quant_ladder.py --stage baseline|int4|vocab|vocab4|static|all
Bench protocol and quality metrics mirror train_distill.py exactly.
"""
import argparse
import gc
import json
import os
import shutil
import string
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import psutil
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS_OUT = HERE / "models" / "edge"
SRC = Path("/home/kamil/projects/here/instruction_extraction/data/instructions_v3.jsonl")
BENCH_PATH = DATA / "edge_bench.jsonl"
GT_PATH = DATA / "edge_gt_fp32.npy"
RESULTS_PATH = DATA / "quant_ladder_results.json"
DISTILL_RESULTS_PATH = DATA / "distill_results.json"
STUDENT_DIR = HERE / "models" / "nav-e5s-distill"
STUDENT_TORCH_EMB = DATA / "nav_e5s_distill_torch_emb.npy"
ONNX_FP32 = MODELS_OUT / "nav_e5s_distill.onnx"
ONNX_INT8 = MODELS_OUT / "nav_e5s_distill_int8.onnx"
ONNX_INT4 = MODELS_OUT / "nav_e5s_distill_int4.onnx"
NBITS_TMP = MODELS_OUT / "nav_e5s_distill_nbits_tmp.onnx"
VOCAB_IDS_JSON = MODELS_OUT / "nav_vocab_kept_ids.json"
VOCAB_FP32 = MODELS_OUT / "nav_e5s_distill_vocab_fp32.onnx"
VOCAB_INT8 = MODELS_OUT / "nav_e5s_distill_vocab8.onnx"
VOCAB_INT4 = MODELS_OUT / "nav_e5s_distill_vocab4.onnx"
STATIC_DIR = MODELS_OUT / "nav_static"

E5_PREFIX = "query: "  # required on ALL texts for e5 models

SEED = 42
MAX_LEN = 256
CPU_THREADS = 16
N_PAIRS = 20_000
K = 10
N_CLUSTERS = 50
N_BS1_TEXTS = 128
SANITY_COS_MIN = 0.9999
INT8_REF_SP = 0.8965  # data/distill_results.json onnx.int8_onnx sp vs GT
INT8_SP_MAX_DELTA = 0.005
STATIC_PCA_DIMS = 300  # whitened PCA dims for static rung (measured best)

# single printable + accented Latin chars (pl/fr/es/de + neighbors) for vocab keeping;
# only ids actually present in the tokenizer vocab are added
ACCENTED = ("àáâãäåæçèéêëìíîïðñòóôõöøùúûüýÿ"
            "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"
            "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
            "ăâîşțĂÂÎŞȚ"
            "ďěĺľňôŕšťůžĎĚĹĽŇÔŔŠŤŮŽ"
            "đģķļ ŽžŠšŒœŸÿ")
EXTRA_CHARS = set(string.printable) | set(ACCENTED)

STAGES = ["baseline", "int4", "vocab", "vocab4", "static"]

PROCESS = psutil.Process(os.getpid())


def rss_mb():
    return PROCESS.memory_info().rss / 1024**2


def l2norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def fmt_bytes(n):
    if n >= 1024**3:
        return f"{n / 1024**3:.2f}GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f}MB"
    return f"{n / 1024:.0f}KB"


def r(x, nd=4):
    v = float(x)
    return round(v, nd) if np.isfinite(v) else None


# ---------- quality metrics (verbatim from train_distill.py / edge_compare.py) ----------

def make_pairs(n, seed=SEED):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, N_PAIRS + 1000)
    j = rng.integers(0, n, N_PAIRS + 1000)
    keep = i != j
    return i[keep][:N_PAIRS], j[keep][:N_PAIRS]


def similarity_spearman(a, b, pairs):
    i, j = pairs
    sa = np.sum(a[i] * a[j], axis=1)
    sb = np.sum(b[i] * b[j], axis=1)
    return r(spearmanr(sa, sb).statistic)


def knn_stats(gt, cand, langs, pairs):
    S_gt, S_c = gt @ gt.T, cand @ cand.T
    np.fill_diagonal(S_gt, -np.inf)
    np.fill_diagonal(S_c, -np.inf)
    ord_gt = np.argsort(-S_gt, axis=1)
    ord_c = np.argsort(-S_c, axis=1)
    n = len(gt)
    jacs, rrs = [], []
    top1_hits = xl_rows = xl_hits = 0
    for row in range(n):
        tg = set(ord_gt[row, :K].tolist())
        tc = set(ord_c[row, :K].tolist())
        jacs.append(len(tg & tc) / len(tg | tc))
        gt_top1 = int(ord_gt[row, 0])
        top_c = ord_c[row, :K]
        hit_pos = np.where(top_c == gt_top1)[0]
        rrs.append(1.0 / (int(hit_pos[0]) + 1) if len(hit_pos) else 0.0)
        hit = int(top_c[0]) == gt_top1
        top1_hits += hit
        if langs[row] != langs[gt_top1]:
            xl_rows += 1
            xl_hits += hit
    return {"knn_jaccard_at10": r(np.mean(jacs)),
            "mrr_at10_of_gt_top1": r(np.mean(rrs)),
            "top1_agreement": r(top1_hits / n),
            "top1_agreement_crosslingual": r(xl_hits / xl_rows) if xl_rows else None,
            "n_crosslingual_top1_texts": xl_rows}


def ari_score(gt, cand):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    lg = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit_predict(gt)
    lc = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit_predict(cand)
    return r(adjusted_rand_score(lg, lc))


def quality_report(gt, cand, langs, pairs):
    q = {"pairwise_spearman_20k_pairs": similarity_spearman(gt, cand, pairs)}
    q.update(knn_stats(gt, cand, langs, pairs))
    q["ari_50_clusters"] = ari_score(gt, cand)
    return q


def quality_flat(q):
    return {"spearman_20k": q["pairwise_spearman_20k_pairs"], "jaccard_at10": q["knn_jaccard_at10"],
            "mrr_at10": q["mrr_at10_of_gt_top1"], "top1": q["top1_agreement"],
            "top1_crosslingual": q["top1_agreement_crosslingual"], "ari_50": q["ari_50_clusters"]}


# ---------- data / bench ----------

def load_bench():
    texts, langs = [], []
    with open(BENCH_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d["text"])
            langs.append(d.get("language", "?"))
    gt = np.load(GT_PATH).astype(np.float32)
    if gt.shape != (len(texts), 1024):
        raise RuntimeError(f"{GT_PATH} shape {gt.shape} does not match bench ({len(texts)}, 1024)")
    return texts, langs, gt


def torch_ref_embeddings(bench_texts):
    import torch
    from sentence_transformers import SentenceTransformer
    if STUDENT_TORCH_EMB.exists():
        emb = np.load(STUDENT_TORCH_EMB).astype(np.float32)
        if emb.shape == (len(bench_texts), 384):
            return emb
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(CPU_THREADS if device == "cpu" else torch.get_num_threads())
    st = SentenceTransformer(str(STUDENT_DIR), device=device)
    st.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(bench_texts), 128):
            enc = st.tokenizer([E5_PREFIX + t for t in bench_texts[i:i + 128]], padding=True,
                               truncation=True, max_length=MAX_LEN, return_tensors="pt").to(st.device)
            out.append(st(enc)["sentence_embedding"].float().cpu().numpy())
    emb = l2norm(np.concatenate(out))
    del st
    if device == "cuda":
        torch.cuda.empty_cache()
    np.save(STUDENT_TORCH_EMB, emb.astype(np.float32))
    return emb


def load_bench_ctx():
    texts, langs, gt = load_bench()
    return {"texts": texts, "langs": langs, "gt": gt, "pairs": make_pairs(len(texts)),
            "torch_ref": torch_ref_embeddings(texts)}


def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(STUDENT_DIR))


# ---------- onnx embedding (train_distill.onnx_embed verbatim + remapped variant) ----------

def onnx_embed(sess, tok, texts, bs=8):
    if E5_PREFIX:
        texts = [E5_PREFIX + t for t in texts]
    outs = []
    names = [i.name for i in sess.get_inputs()]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        h = next(o for o in sess.run(None, feed) if o.ndim == 3)
        mask = enc["attention_mask"].astype(np.float32)
        v = (h * mask[:, :, None]).sum(1) / np.maximum(mask.sum(1)[:, None], 1.0)
        outs.append(v.astype(np.float32))
    return l2norm(np.concatenate(outs))


def onnx_embed_remapped(sess, tok, texts, lut, bs=8):
    """Same protocol as onnx_embed but tokenizes with the FULL original tokenizer and
    remaps ids old->new via lut (unseen ids -> unk_new_id) before the session run."""
    if E5_PREFIX:
        texts = [E5_PREFIX + t for t in texts]
    outs = []
    names = [i.name for i in sess.get_inputs()]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="np")
        feed = {"input_ids": lut[enc["input_ids"].astype(np.int64)],
                "attention_mask": enc["attention_mask"].astype(np.int64)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        h = next(o for o in sess.run(None, feed) if o.ndim == 3)
        mask = enc["attention_mask"].astype(np.float32)
        v = (h * mask[:, :, None]).sum(1) / np.maximum(mask.sum(1)[:, None], 1.0)
        outs.append(v.astype(np.float32))
    return l2norm(np.concatenate(outs))


def onnx_embeddings(path, tok, texts, lut=None):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    try:
        if lut is None:
            return onnx_embed(sess, tok, texts)
        return onnx_embed_remapped(sess, tok, texts, lut)
    finally:
        del sess
        gc.collect()


# ---------- bench protocol (mirrors train_distill.bench_session) ----------

def ort_factory(path, threads=CPU_THREADS):
    import onnxruntime as ort

    def make_model():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return make_model


def bench_session(make_model, embed, texts, n_bs1=N_BS1_TEXTS):
    gc.collect()
    pre = rss_mb()
    t0 = time.perf_counter()
    model = make_model()
    load_s = time.perf_counter() - t0
    peak = max(rss_mb(), pre)
    embed(model, texts[:16], bs=16)  # warmup
    peak = max(peak, rss_mb())

    lat = []
    t0 = time.perf_counter()
    for t in texts[:n_bs1]:
        t1 = time.perf_counter()
        embed(model, [t], bs=1)
        lat.append((time.perf_counter() - t1) * 1000)
    bs1_dt = time.perf_counter() - t0
    peak = max(peak, rss_mb())

    perf = {"load_s": round(load_s, 2),
            "p50_ms": r(np.median(lat), 2), "p95_ms": r(np.percentile(lat, 95), 2),
            "bs1_texts_s": round(len(lat) / bs1_dt, 1)}
    for bsz in (8, 32):
        t0 = time.perf_counter()
        embed(model, texts, bs=bsz)
        dt = time.perf_counter() - t0
        peak = max(peak, rss_mb())
        perf[f"tps_bs{bsz}"] = round(len(texts) / dt, 1)
        print(f"  bs{bsz}: {len(texts) / dt:.1f} texts/s", flush=True)
    perf["peak_rss_mb"] = round(peak)
    perf["rss_delta_mb"] = round(peak - pre)
    del model
    gc.collect()
    return perf


def ort_bench(path, tok, texts, lut=None):
    def embed(sess, batch, bs):
        if lut is None:
            return onnx_embed(sess, tok, batch, bs=bs)
        return onnx_embed_remapped(sess, tok, batch, lut=lut, bs=bs)
    return bench_session(ort_factory(path), embed, texts)


# ---------- artifacts / quantize helpers ----------

def artifact_files(path):
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    return sorted(p for p in MODELS_OUT.glob(path.name + "*") if p.is_file())


def files_entry(path):
    files = artifact_files(path)
    total = sum(f.stat().st_size for f in files)
    return [{"path": str(f), "bytes": f.stat().st_size} for f in files], total


def clear_stale(path):
    """Delete stale outputs BEFORE quantizing/writing (ORT quantizer appends to
    existing external data files)."""
    if path.is_dir():
        if path.exists():
            shutil.rmtree(path)
    else:
        for p in MODELS_OUT.glob(path.name + "*"):
            p.unlink()


def artifact_bytes(out_path):
    return sum(f.stat().st_size for f in sorted(out_path.parent.glob(out_path.name + "*")))


def quantize_dynamic_to(fp32_path, out_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    clear_stale(out_path)
    try:
        quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
        external = False
    except Exception:
        quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8,
                         use_external_data_format=True)
        external = True
    return artifact_bytes(out_path), external


def nbits4_quantize(fp32_path, out_path):
    """MatMulNBits (block 64, 4-bit, accuracy_level 4). Default op_types_to_quantize
    is {MatMul} so the embedding Gather stays fp32 here and becomes int8 in the
    following quantize_dynamic step."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
    clear_stale(out_path)
    model = onnx.load(str(fp32_path))
    try:
        config = DefaultWeightOnlyQuantConfig(block_size=64, bits=4, accuracy_level=4)
    except TypeError:
        config = DefaultWeightOnlyQuantConfig(block_size=64, bits=4)
    quant = MatMulNBitsQuantizer(model=model, algo_config=config)
    quant.process()
    quant.model.save_model_to_file(str(out_path), use_external_data_format=False)
    return artifact_bytes(out_path)


def nbits_then_dynamic(fp32_path, out_path, tag):
    """NBits first (MatMul -> MatMulNBits), then quantize_dynamic on the result
    (int8s the remaining fp32 weights incl. the embedding Gather, skips NBits nodes).
    Falls back to NBits-only (embeddings fp32) if dynamic quantization errors."""
    clear_stale(NBITS_TMP)
    nbits_bytes = nbits4_quantize(fp32_path, NBITS_TMP)
    print(f"  [{tag}] MatMulNBits intermediate: {fmt_bytes(nbits_bytes)}", flush=True)
    note, dynamic_failed = None, False
    try:
        total_bytes, external = quantize_dynamic_to(NBITS_TMP, out_path)
    except Exception as e:
        dynamic_failed = True
        note = f"quantize_dynamic failed ({e}); kept NBits-only output (embeddings fp32)"
        print(f"  [{tag}] WARNING: {note}", flush=True)
        clear_stale(out_path)
        shutil.copyfile(NBITS_TMP, out_path)
        total_bytes, external = artifact_bytes(out_path), False
    for p in NBITS_TMP.parent.glob(NBITS_TMP.name + "*"):
        p.unlink()
    return total_bytes, external, dynamic_failed, note


def finish_rung(entry, paths, perf, emb, ctx):
    files, total = [], 0
    for p in paths:
        fl, t = files_entry(p)
        files.extend(fl)
        total += t
    if emb.shape[1] == ctx["torch_ref"].shape[1]:
        cos = np.sum(emb * ctx["torch_ref"], axis=1)
        cos_entry = {"cos_vs_torch": r(cos.mean(), 6), "cos_vs_torch_min": r(cos.min(), 6)}
    else:
        cos_entry = {"cos_vs_torch": None,
                     "cos_vs_torch_note": f"n/a: {emb.shape[1]}d rung vs {ctx['torch_ref'].shape[1]}d torch (PCA-whitened space)"}
    entry.update({"files": files, "total_mb": round(total / 1024**2, 1), **perf,
                  **cos_entry,
                  "quality": quality_flat(quality_report(ctx["gt"], emb, ctx["langs"], ctx["pairs"]))})
    return None


# ---------- vocab pruning ----------

def load_corpus_texts():
    texts = []
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t)
    return texts


def build_kept_ids(tok):
    """Ids used by the corpus (SAME preprocessing as inference: e5 prefix + truncation)
    union specials, single printable ASCII chars and accented Latin chars present in
    the vocab. Sorted ascending -> stable new id = rank in this order."""
    vocab = tok.get_vocab()
    corpus = load_corpus_texts()
    ids = set()
    chunk = 2048
    for i in range(0, len(corpus), chunk):
        enc = tok([E5_PREFIX + t for t in corpus[i:i + chunk]], padding=False,
                  truncation=True, max_length=MAX_LEN)
        for row in enc["input_ids"]:
            ids.update(row)
    n_corpus = len(ids)
    ids.update(tok.all_special_ids)
    n_special = len(ids)
    n_char = 0
    for ch in EXTRA_CHARS:
        tid = vocab.get(ch)
        if tid is not None:
            ids.add(tid)
            n_char += 1
    kept = sorted(ids)
    print(f"[vocab] corpus ids {n_corpus} + specials -> {n_special} + single chars ({n_char}) "
          f"= kept {len(kept)} of {len(vocab)}", flush=True)
    return kept, {"n_corpus_ids": n_corpus, "n_after_specials": n_special,
                  "n_char_tokens_added": n_char, "n_corpus_texts": len(corpus)}


def get_kept_ids(tok):
    if VOCAB_IDS_JSON.exists():
        d = json.loads(VOCAB_IDS_JSON.read_text())
        print(f"[vocab] reused {len(d['kept_old_ids'])} kept ids from {VOCAB_IDS_JSON.name}", flush=True)
        return d
    kept, stats = build_kept_ids(tok)
    unk_old = tok.get_vocab()["<unk>"]
    unk_new = kept.index(unk_old) if unk_old in kept else 0
    d = {"kept_old_ids": kept, "orig_vocab": len(tok), "unk_old_id": unk_old,
         "unk_new_id": unk_new, **stats}
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    VOCAB_IDS_JSON.write_text(json.dumps(d, indent=0))
    print(f"[vocab] saved {VOCAB_IDS_JSON} (unk old {unk_old} -> new {unk_new})", flush=True)
    return d


def make_lut(kept_data):
    lut = np.full(kept_data["orig_vocab"], kept_data["unk_new_id"], dtype=np.int64)
    lut[np.asarray(kept_data["kept_old_ids"], dtype=np.int64)] = np.arange(len(kept_data["kept_old_ids"]))
    return lut


# ---------- stage baseline ----------

def stage_baseline(results, ctx):
    entry = results["baseline"]
    tok = get_tok()
    print(f"[baseline] bench {ONNX_INT8.name} (this protocol)", flush=True)
    perf = ort_bench(ONNX_INT8, tok, ctx["texts"])
    emb = onnx_embeddings(ONNX_INT8, tok, ctx["texts"])
    finish_rung(entry, [ONNX_INT8], perf, emb, ctx)
    if DISTILL_RESULTS_PATH.exists():
        dr = json.loads(DISTILL_RESULTS_PATH.read_text())
        onx = dr.get("onnx") or {}
        cfg = next(iter(onx.values()), {}) if isinstance(onx, dict) else {}
        ref = cfg.get("int8_onnx")
        if ref:
            entry["reference_distill_results"] = ref
            print(f"[baseline] reference (train_distill): size={ref.get('size_mb')}MB "
                  f"p50={(ref.get('bs1') or {}).get('latency_ms', {}).get('p50')}ms "
                  f"bs8={(ref.get('bs8') or {}).get('texts_s')}t/s "
                  f"sp={(ref.get('quality_vs_gt') or {}).get('pairwise_spearman_20k_pairs')}", flush=True)
    print(f"[baseline] this run: p50={perf['p50_ms']}ms bs8={perf['tps_bs8']}t/s "
          f"cos_torch={entry['cos_vs_torch']} sp={entry['quality']['spearman_20k']}", flush=True)


# ---------- stage int4 ----------

def stage_int4(results, ctx):
    if not ONNX_FP32.exists():
        raise RuntimeError(f"{ONNX_FP32} missing; run train_distill.py --stage onnx first")
    tok = get_tok()
    total_bytes, external, dynamic_failed, note = nbits_then_dynamic(ONNX_FP32, ONNX_INT4, "int4")
    print(f"[int4] -> {ONNX_INT4.name} ({fmt_bytes(total_bytes)}, external={external})", flush=True)
    perf = ort_bench(ONNX_INT4, tok, ctx["texts"])
    emb = onnx_embeddings(ONNX_INT4, tok, ctx["texts"])
    entry = results["int4"]
    entry.update({"nbits_matmul_int4_then_dynamic_int8": not dynamic_failed,
                  "external_data": external})
    if note:
        entry["note"] = note
    finish_rung(entry, [ONNX_INT4], perf, emb, ctx)
    print(f"[int4] {entry['total_mb']}MB p50={perf['p50_ms']}ms bs8={perf['tps_bs8']}t/s "
          f"cos_torch={entry['cos_vs_torch']} sp={entry['quality']['spearman_20k']}", flush=True)


# ---------- stage vocab (prune + fp32 export + gate + int8) ----------

def export_pruned_onnx(st, out_path, lut):
    """Mirrors train_distill.export_onnx (wrapper, dynamo=False, opset 17,
    .eval().to('cpu')); probe ids are remapped so tracing never indexes rows
    beyond the pruned embedding table."""
    import torch

    class OnnxWrap(torch.nn.Module):
        def __init__(self, auto_model):
            super().__init__()
            self.m = auto_model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    bench_texts, _, _ = load_bench()
    probe = st.tokenizer([E5_PREFIX + t for t in bench_texts[:4]], padding=True,
                         truncation=True, max_length=MAX_LEN, return_tensors="pt")
    ids = torch.from_numpy(lut[probe["input_ids"].numpy()].copy())
    tti = torch.zeros_like(ids)
    wrap = OnnxWrap(st[0].auto_model).eval().to("cpu")
    with torch.no_grad():
        torch.onnx.export(
            wrap, (ids, probe["attention_mask"], tti), str(out_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "attention_mask": {0: "batch", 1: "seq"},
                          "token_type_ids": {0: "batch", 1: "seq"},
                          "last_hidden_state": {0: "batch", 1: "seq"}},
            opset_version=17, dynamo=False, do_constant_folding=True)


def prune_and_export(kept_data, out_path):
    """Overwrite word embedding rows with the kept old rows (sorted ascending);
    position/token_type embeddings are untouched by resize_token_embeddings."""
    import torch
    from sentence_transformers import SentenceTransformer
    clear_stale(out_path)
    st = SentenceTransformer(str(STUDENT_DIR), device="cpu")
    st.eval()
    auto = st[0].auto_model
    old_w = auto.get_input_embeddings().weight.data.clone()
    shapes_before = (auto.embeddings.position_embeddings.weight.shape,
                     auto.embeddings.token_type_embeddings.weight.shape)
    kept = torch.tensor(kept_data["kept_old_ids"], dtype=torch.long)
    auto.resize_token_embeddings(len(kept))
    emb = auto.get_input_embeddings()
    if tuple(emb.weight.shape) != (len(kept), old_w.shape[1]):
        raise RuntimeError(f"resize gave {tuple(emb.weight.shape)}, expected {(len(kept), old_w.shape[1])}")
    if auto.embeddings.position_embeddings.weight.shape != shapes_before[0] or \
            auto.embeddings.token_type_embeddings.weight.shape != shapes_before[1]:
        raise RuntimeError("resize_token_embeddings touched position/token_type embeddings")
    with torch.no_grad():
        emb.weight.copy_(old_w[kept])
    if not torch.equal(emb.weight.data, old_w[kept]):
        raise RuntimeError("pruned word embeddings do not match kept old rows")
    lut = make_lut(kept_data)
    export_pruned_onnx(st, out_path, lut)
    del st, old_w
    gc.collect()


def ensure_pruned_fp32(kept_data):
    if VOCAB_FP32.exists():
        return
    t0 = time.perf_counter()
    prune_and_export(kept_data, VOCAB_FP32)
    print(f"[vocab] pruned fp32 -> {VOCAB_FP32.name} ({fmt_bytes(VOCAB_FP32.stat().st_size)}) "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)


def stage_vocab(results, ctx):
    tok = get_tok()
    kept_data = get_kept_ids(tok)
    lut = make_lut(kept_data)
    ensure_pruned_fp32(kept_data)

    emb = onnx_embeddings(VOCAB_FP32, tok, ctx["texts"], lut=lut)
    cos = np.sum(emb * ctx["torch_ref"], axis=1)
    sanity = r(cos.mean(), 6)
    print(f"[vocab] sanity gate cos-vs-torch = {sanity} (min {r(cos.min(), 6)})", flush=True)

    if sanity < SANITY_COS_MIN:
        results["vocab"].update({"kept_vocab_size": len(kept_data["kept_old_ids"]),
                                 "unk_new_id": kept_data["unk_new_id"], "sanity_cos": sanity,
                                 "error": f"sanity gate failed: cos {sanity} < {SANITY_COS_MIN}"})
        raise RuntimeError(f"sanity gate failed: cos-vs-torch {sanity} < {SANITY_COS_MIN}; rung aborted")

    total_bytes, external = quantize_dynamic_to(VOCAB_FP32, VOCAB_INT8)
    print(f"[vocab] -> {VOCAB_INT8.name} ({fmt_bytes(total_bytes)}, external={external})", flush=True)
    perf = ort_bench(VOCAB_INT8, tok, ctx["texts"], lut=lut)
    emb = onnx_embeddings(VOCAB_INT8, tok, ctx["texts"], lut=lut)
    entry = results["vocab"]
    entry.update({"kept_vocab_size": len(kept_data["kept_old_ids"]),
                  "unk_new_id": kept_data["unk_new_id"], "sanity_cos": sanity,
                  "orig_vocab": kept_data["orig_vocab"],
                  "quant": "pure dynamic int8", "external_data": external,
                  "fp32_intermediate_mb": round(VOCAB_FP32.stat().st_size / 1024**2, 1),
                  "fp32_intermediate_note": "build intermediate for vocab4; not a deploy artifact"})
    finish_rung(entry, [VOCAB_INT8], perf, emb, ctx)
    sp = entry["quality"]["spearman_20k"]
    entry["sp_delta_vs_int8_ref"] = r(abs(sp - INT8_REF_SP), 4)
    entry["sp_within_acceptance"] = bool(entry["sp_delta_vs_int8_ref"] <= INT8_SP_MAX_DELTA)
    print(f"[vocab] {entry['total_mb']}MB p50={perf['p50_ms']}ms bs8={perf['tps_bs8']}t/s "
          f"sp={sp} (delta vs int8 ref {entry['sp_delta_vs_int8_ref']})", flush=True)


# ---------- stage vocab4 ----------

def stage_vocab4(results, ctx):
    tok = get_tok()
    kept_data = get_kept_ids(tok)
    lut = make_lut(kept_data)
    ensure_pruned_fp32(kept_data)

    total_bytes, external, dynamic_failed, note = nbits_then_dynamic(VOCAB_FP32, VOCAB_INT4, "vocab4")
    print(f"[vocab4] -> {VOCAB_INT4.name} ({fmt_bytes(total_bytes)}, external={external})", flush=True)
    perf = ort_bench(VOCAB_INT4, tok, ctx["texts"], lut=lut)
    emb = onnx_embeddings(VOCAB_INT4, tok, ctx["texts"], lut=lut)
    entry = results["vocab4"]
    entry.update({"kept_vocab_size": len(kept_data["kept_old_ids"]),
                  "unk_new_id": kept_data["unk_new_id"], "sanity_cos": None,
                  "nbits_matmul_int4_then_dynamic_int8": not dynamic_failed,
                  "external_data": external})
    if note:
        entry["note"] = note
    finish_rung(entry, [VOCAB_INT4], perf, emb, ctx)
    print(f"[vocab4] {entry['total_mb']}MB p50={perf['p50_ms']}ms bs8={perf['tps_bs8']}t/s "
          f"cos_torch={entry['cos_vs_torch']} sp={entry['quality']['spearman_20k']}", flush=True)


# ---------- stage static (model2vec-style lookup model) ----------

def static_distill_full(device="cuda"):
    """Replicates model2vec create_embeddings (MEAN pooling) faithfully: each vocab
    token is re-encoded WITH special tokens (<s> tok </s>) and run through the
    transformer; its static vector is the attention-masked mean of the hidden
    states -- NOT a bare length-1 forward (which is OOD for this teacher)."""
    import inspect

    import torch
    from sentence_transformers import SentenceTransformer
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tok = get_tok()
    st = SentenceTransformer(str(STUDENT_DIR), device=device)
    st.eval()
    auto = st[0].auto_model
    n_model, dim = auto.get_input_embeddings().weight.shape  # 250037 (padded)
    n_tok = len(tok)  # 250002: ids the tokenizer can actually emit
    has_tti = "token_type_ids" in inspect.getfullargspec(auto.forward).args

    bos = tok.bos_token_id if tok.bos_token_id is not None else 0
    eos = tok.eos_token_id if tok.eos_token_id is not None else 2
    rng = np.random.default_rng(SEED)
    sample = rng.choice(n_tok, size=2000, replace=False)
    fast_ok = all(
        tok(tok.convert_ids_to_tokens(int(i)), add_special_tokens=True)["input_ids"] == [bos, int(i), eos]
        for i in sample
    )
    if fast_ok:
        seqs = [[bos, i, eos] for i in range(n_tok)]
    else:
        print("[static] round-trip sample failed; falling back to per-token re-encode", flush=True)
        seqs = [tok(tok.convert_ids_to_tokens(i), add_special_tokens=True)["input_ids"]
                for i in range(n_tok)]

    # rows beyond the tokenizer vocab (padding rows 250002..) are unreachable: zeros
    out = np.zeros((n_model, dim), dtype=np.float32)
    bs = 256
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, n_tok, bs):
            e = min(s + bs, n_tok)
            batch = seqs[s:e]
            maxlen = max(len(x) for x in batch)
            ids = torch.full((e - s, maxlen), tok.pad_token_id or 1, dtype=torch.long, device=device)
            mask = torch.zeros((e - s, maxlen), dtype=torch.long, device=device)
            interior = torch.zeros((e - s, maxlen), dtype=torch.bool, device=device)
            for j, x in enumerate(batch):
                ids[j, :len(x)] = torch.tensor(x, dtype=torch.long, device=device)
                mask[j, :len(x)] = 1
                interior[j, 1:len(x) - 1] = True  # exclude bos/eos: token position(s) only
            feed = {"input_ids": ids, "attention_mask": mask}
            if has_tti:
                feed["token_type_ids"] = torch.zeros_like(ids)
            h = auto(**feed).last_hidden_state
            m = interior.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
            out[s:e] = pooled.float().cpu().numpy()
            if (s // bs) % 200 == 0:
                print(f"  static distill {e}/{n_tok}", flush=True)
    dt = time.perf_counter() - t0
    print(f"[static] distilled full matrix {out.shape} in {dt:.1f}s (fast_path={fast_ok}, "
          f"token-position pooling)", flush=True)
    del st, auto
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def build_static_model(kept_data):
    """Token-position static matrix -> PCA-300 whitening (fit on kept rows) -> slice to
    kept ids; unk row = static <unk> vector. The full original tokenizer is kept and
    token_mapping (model2vec's vocab-slice mechanism) maps old id -> row, with
    non-kept ids pointing at the trailing unk row. PCA whitening measured best:
    sp 0.452 vs 0.365 unwhitened / 0.198 SIF+PCA / 0.063 mean-incl-specials."""
    from model2vec import StaticModel
    from sklearn.decomposition import PCA
    full = static_distill_full()
    kept = np.asarray(kept_data["kept_old_ids"], dtype=np.int64)
    pca = PCA(n_components=min(STATIC_PCA_DIMS, full.shape[1], len(kept)), whiten=True)
    pca.fit(full[kept])
    w = pca.transform(full).astype(np.float32)
    vectors = np.concatenate([w[kept], w[kept_data["unk_old_id"]][None, :]], axis=0).astype(np.float32)
    mapping = np.full(kept_data["orig_vocab"], len(kept), dtype=np.int64)
    mapping[kept] = np.arange(len(kept), dtype=np.int64)
    bt = get_tok().backend_tokenizer
    # CRITICAL: the e5 tokenizer.json carries a BatchLongest padding config; left
    # enabled, model2vec's batched tokenize() pads every text to the batch max and
    # _encode_batch's mean pools over the pad rows -> garbage embeddings (sp -0.10).
    bt.no_padding()
    model = StaticModel(vectors=vectors, tokenizer=bt,
                        config={"normalize": True}, normalize=True,
                        token_mapping=mapping, base_model_name="nav-e5s-distill-static")
    return model


def static_embed(model, texts, bs):
    # no e5 "query: " prefix: static vectors are prefix-free (model2vec convention);
    # a shared prefix would add a constant component to every embedding
    model.tokenizer.no_padding()  # belt-and-braces vs tokenizer.json padding config
    return model.encode(texts, batch_size=bs)


def stage_static(results, ctx):
    from model2vec import StaticModel
    note = None
    try:
        from model2vec.distill import distill  # noqa: F401
        method = "local faithful replication of model2vec create_embeddings (specials-wrapped tokens, mean pool)"
        note = "model2vec.distill importable but its TokenizerModel pipeline not wired; used local replication"
    except ImportError as e:
        method = "local faithful replication of model2vec create_embeddings (specials-wrapped tokens, mean pool)"
        note = f"model2vec.distill unavailable: {e}"
        print(f"[static] {note}", flush=True)

    tok = get_tok()
    kept_data = get_kept_ids(tok)
    clear_stale(STATIC_DIR)
    model = build_static_model(kept_data)
    try:
        from model2vec import quantize_model
        model = quantize_model(model, quantize_to="int8")
        quant_note = "vectors int8 (model2vec quantize_model)"
    except Exception as e:
        quant_note = f"vectors fp32 (int8 quantize failed: {e})"
    print(f"[static] {quant_note}", flush=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(STATIC_DIR), model_name="nav-e5s-distill-static")
    print(f"[static] saved -> {STATIC_DIR} ({fmt_bytes(sum(f.stat().st_size for f in STATIC_DIR.rglob('*') if f.is_file()))})",
          flush=True)
    del model
    gc.collect()

    loaded = StaticModel.from_pretrained(str(STATIC_DIR))
    perf = bench_session(lambda: StaticModel.from_pretrained(str(STATIC_DIR)),
                         lambda m, batch, bs: static_embed(m, batch, bs), ctx["texts"])
    emb = static_embed(loaded, ctx["texts"], bs=64).astype(np.float32)
    del loaded
    gc.collect()

    entry = results["static"]
    entry.update({"method": method, "normalize": True, "sif_weights": None,
                  "pca_dims_whitened": STATIC_PCA_DIMS, "quant_note": quant_note,
                  "vectors_dtype": "float32", "kept_vocab_size": len(kept_data["kept_old_ids"]),
                  "unk_new_id": kept_data["unk_new_id"]})
    if note:
        entry["note"] = note
    finish_rung(entry, [STATIC_DIR], perf, emb, ctx)
    print(f"[static] {entry['total_mb']}MB p50={perf['p50_ms']}ms bs8={perf['tps_bs8']}t/s "
          f"cos_torch={entry['cos_vs_torch']} sp={entry['quality']['spearman_20k']}", flush=True)


# ---------- summary ----------

def print_summary(results):
    ref = (results["baseline"].get("reference_distill_results") or {})

    def refget(path, default=None):
        cur = ref
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def get(rung, path):
        cur = results.get(rung) or {}
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    cols = [("baseline(ref)", refget, True)]
    cols += [(name, lambda p, n=name: get(n, p), False)
             for name in ("baseline", "int4", "vocab", "vocab4", "static")]
    rows = [
        ("size_mb", "total_mb", "size_mb"),
        ("load_s", "load_s", "load_s"),
        ("p50_ms", "p50_ms", "bs1.latency_ms.p50"),
        ("p95_ms", "p95_ms", "bs1.latency_ms.p95"),
        ("tps_bs8", "tps_bs8", "bs8.texts_s"),
        ("tps_bs32", "tps_bs32", "bs32.texts_s"),
        ("rss_delta_mb", "rss_delta_mb", "rss_delta_mb"),
        ("cos_vs_torch", "cos_vs_torch", "cos_vs_torch_fp32.mean"),
        ("sp_20k", "quality.spearman_20k", "quality_vs_gt.pairwise_spearman_20k_pairs"),
        ("jac@10", "quality.jaccard_at10", "quality_vs_gt.knn_jaccard_at10"),
        ("mrr@10", "quality.mrr_at10", "quality_vs_gt.mrr_at10_of_gt_top1"),
        ("top1", "quality.top1", "quality_vs_gt.top1_agreement"),
        ("top1_xl", "quality.top1_crosslingual", "quality_vs_gt.top1_agreement_crosslingual"),
        ("ari(50)", "quality.ari_50", "quality_vs_gt.ari_50_clusters"),
    ]
    print("\n== quantization ladder (vs GT = bge-m3 fp32; cos vs torch fp32 distill) ==")
    header = f"{'metric':<14}" + "".join(f"{name:>15}" for name, _, _ in cols)
    print(header)
    for label, rung_path, ref_path in rows:
        cells = []
        for _, getter, is_ref in cols:
            v = getter(ref_path if is_ref else rung_path)
            cells.append(f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v)))
        print(f"{label:<14}" + "".join(f"{c:>15}" for c in cells))
    errs = results.get("errors") or {}
    if errs:
        print(f"errors in: {', '.join(errs)}")


# ---------- main ----------

def load_results():
    if RESULTS_PATH.exists():
        try:
            res = json.loads(RESULTS_PATH.read_text())
        except Exception as e:
            print(f"warning: could not parse existing results ({e}); starting fresh")
            res = {}
    else:
        res = {}
    for k in ("meta", "baseline", "int4", "vocab", "vocab4", "static", "errors"):
        res.setdefault(k, {})
    return res


def main():
    np.random.seed(SEED)

    ap = argparse.ArgumentParser(description="Quantization ladder for nav-e5s-distill: baseline/int4/vocab/vocab4/static")
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    import transformers

    ctx = load_bench_ctx()

    results = load_results()
    results["meta"].update({
        "script": "quant_ladder.py", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED, "onnxruntime": ort.__version__, "onnx": onnx.__version__,
        "transformers": transformers.__version__, "cuda_device": None,
        "hf_hub_offline": True, "student_dir": str(STUDENT_DIR),
        "bench_texts": len(ctx["texts"]), "gt": str(GT_PATH),
    })
    try:
        import torch
        import model2vec
        results["meta"].update({"torch": torch.__version__, "model2vec": model2vec.__version__,
                                "cuda_device": torch.cuda.get_device_name(0)
                                if torch.cuda.is_available() else None})
    except Exception:
        pass

    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        t0 = time.perf_counter()
        try:
            if stage == "baseline":
                stage_baseline(results, ctx)
            elif stage == "int4":
                stage_int4(results, ctx)
            elif stage == "vocab":
                stage_vocab(results, ctx)
            elif stage == "vocab4":
                stage_vocab4(results, ctx)
            elif stage == "static":
                stage_static(results, ctx)
            results["errors"].pop(stage, None)
        except Exception:
            results["errors"][stage] = traceback.format_exc()[-3000:]
            print(f"[{stage}] ERROR: {traceback.format_exc(limit=3)}", flush=True)
        results["meta"][f"{stage}_wall_s"] = round(time.perf_counter() - t0, 1)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[{stage}] saved -> {RESULTS_PATH}", flush=True)

    print_summary(results)


if __name__ == "__main__":
    main()
