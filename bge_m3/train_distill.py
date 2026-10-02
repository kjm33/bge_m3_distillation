"""Distill BAAI/bge-m3 (teacher, 1024-d CLS) into intfloat/multilingual-e5-small
(student, 384-d mean-pool) for car navigation instructions, then export ONNX int8
for the CPU-only edge target.

Stages (runnable separately, results merged into data/distill_results.json):
  teacher - embed deduped train pool (corpus minus bench texts) with bge-m3
            fp32 CUDA CLS pooling (method identical to edge_gt_fp32.npy)
  train   - manual PyTorch loop, Matryoshka KL-matrix distillation over
            hard-neighbor batches (teacher top-64 NN); per-epoch eval on the
            512-text bench vs GT; best checkpoint saved to models/nav-e5s-distill
  onnx    - export saved model to ONNX fp32 + dynamic int8, CPU bench
            (load/latency/throughput/RSS) + quality vs GT (mirrors edge_compare)

Run from bge_m3/:  .venv/bin/python train_distill.py --stage teacher|train|onnx|all
"""
import argparse
import gc
import json
import os
import random
import time
import traceback
import unicodedata
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import psutil
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS_OUT = HERE / "models" / "edge"
SRC = DATA / "instructions_v3.jsonl"
BENCH_PATH = DATA / "edge_bench.jsonl"
GT_PATH = DATA / "edge_gt_fp32.npy"
RESULTS_PATH = DATA / "distill_results.json"
TEACHER_NPY = DATA / "teacher_full_bge_m3.npy"
TRAIN_TEXTS = DATA / "train_texts.jsonl"
STUDENT_DIR = HERE / "models" / "nav-e5s-distill"
STUDENT_TORCH_EMB = DATA / "nav_e5s_distill_torch_emb.npy"
ONNX_FP32 = MODELS_OUT / "nav_e5s_distill.onnx"
ONNX_INT8 = MODELS_OUT / "nav_e5s_distill_int8.onnx"

TEACHER_REPO = "BAAI/bge-m3"
STUDENT_REPO = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "  # required on ALL texts for e5 models

SEED = 42
MAX_LEN = 256
TEACHER_BATCH = 256
TOPK = 64
WINDOW = 7            # neighbors per anchor group (group = 1 + WINDOW)
DEFAULT_TAU = 0.03
DEFAULT_BS = 64
DEFAULT_EPOCHS = 8
DEFAULT_LR = 2e-5
DEFAULT_DIMS = [64, 128, 256, 384]
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
GRAD_CLIP = 1.0
CPU_THREADS = 16
N_PAIRS = 20_000
K = 10
N_CLUSTERS = 50
N_BS1_TEXTS = 128

# edge_compare.py section-6 reference numbers for stock e5-small (printed side by side)
STOCK_REF = {
    "torch_fp32": {"pairwise_spearman_20k_pairs": 0.572, "knn_jaccard_at10": 0.356,
                   "mrr_at10_of_gt_top1": 0.551, "top1_agreement": 0.424,
                   "top1_agreement_crosslingual": 0.012, "ari_50_clusters": 0.252},
    "onnx_int8": {"size_mb": 112.6, "bs8_texts_s": 230, "p50_ms": 5.1,
                  "cos_vs_torch": 0.988, "knn_jaccard_at10_vs_gt": 0.345},
}

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


def nfc_strip(t):
    return unicodedata.normalize("NFC", t).strip()


# ---------- quality metrics (verbatim methodology from edge_compare.py) ----------

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


# ---------- data pool ----------

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


def build_pool(bench_texts):
    """Corpus minus bench texts (NFC+strip equality); keeps ALL other texts incl. JSON artifacts."""
    bench_keys = {nfc_strip(t) for t in bench_texts}
    seen = set()
    texts, langs = [], []
    n_total = n_bench_excl = n_dupe = 0
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("text")
            if not isinstance(t, str) or not t.strip():
                continue
            n_total += 1
            k = nfc_strip(t)
            if k in bench_keys:
                n_bench_excl += 1
                continue
            if k in seen:
                n_dupe += 1
                continue
            seen.add(k)
            texts.append(t)
            langs.append(d.get("language", "?"))
    info = {"source": str(SRC), "n_source_rows": n_total, "excluded_bench_overlap": n_bench_excl,
            "excluded_duplicates": n_dupe, "n_pool": len(texts)}
    print(f"[pool] {n_total} rows -> {len(texts)} texts "
          f"(excluded {n_bench_excl} bench overlaps, {n_dupe} dupes)", flush=True)
    return texts, langs, info


def ensure_train_pool(bench_texts):
    if TRAIN_TEXTS.exists():
        texts, langs = [], []
        with open(TRAIN_TEXTS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                texts.append(d["text"])
                langs.append(d.get("language", "?"))
        print(f"[pool] reused {len(texts)} texts from {TRAIN_TEXTS}", flush=True)
        return texts, langs
    texts, langs, info = build_pool(bench_texts)
    TRAIN_TEXTS.parent.mkdir(exist_ok=True)
    with open(TRAIN_TEXTS, "w") as f:
        for t, l in zip(texts, langs):
            f.write(json.dumps({"text": t, "language": l}, ensure_ascii=False) + "\n")
    return texts, langs


# ---------- stage 1: teacher ----------

def embed_teacher(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("teacher embedding needs CUDA (bge-m3 fp32)")
    tok = AutoTokenizer.from_pretrained(TEACHER_REPO)
    model = AutoModel.from_pretrained(TEACHER_REPO).to("cuda").eval()
    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for b, i in enumerate(range(0, len(texts), TEACHER_BATCH)):
            enc = tok(texts[i:i + TEACHER_BATCH], padding=True, truncation=True,
                      max_length=8192, return_tensors="pt").to("cuda")
            out.append(model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
            if (b + 1) % 10 == 0:
                dt = time.perf_counter() - t0
                print(f"  teacher {min(i + TEACHER_BATCH, len(texts))}/{len(texts)} "
                      f"({len(out) * TEACHER_BATCH / dt:.0f} texts/s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return l2norm(np.concatenate(out).astype(np.float32))


def stage_teacher(results):
    import torch
    t0 = time.perf_counter()
    bench_texts, _, _ = load_bench()
    if TEACHER_NPY.exists() and TRAIN_TEXTS.exists():
        n_jsonl = sum(1 for line in open(TRAIN_TEXTS) if line.strip())
        emb = np.load(TEACHER_NPY)
        if emb.shape[0] == n_jsonl:
            print(f"[teacher] up to date: {TEACHER_NPY} rows={emb.shape[0]}, skipped", flush=True)
            results["teacher"].setdefault("skipped", True)
            return
        print(f"[teacher] stale files (npy rows={emb.shape[0]} != jsonl rows={n_jsonl}), rebuilding", flush=True)
    texts, _ = ensure_train_pool(bench_texts)
    emb = embed_teacher(texts)
    np.save(TEACHER_NPY, emb)
    results["teacher"] = {"model": TEACHER_REPO, "pooling": "cls", "dtype": "fp32", "l2_normalized": True,
                          "device": "cuda", "batch": TEACHER_BATCH, "max_length": 8192,
                          "n": int(emb.shape[0]), "dim": int(emb.shape[1]),
                          "npy": str(TEACHER_NPY), "texts_jsonl": str(TRAIN_TEXTS),
                          "excluded_bench_overlap": info_count(texts, bench_texts),
                          "wall_s": round(time.perf_counter() - t0, 1),
                          "torch": torch.__version__,
                          "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(f"[teacher] {emb.shape} -> {TEACHER_NPY.name} in {results['teacher']['wall_s']}s", flush=True)


def info_count(texts, bench_texts):
    bench_keys = {nfc_strip(t) for t in bench_texts}
    return sum(1 for t in texts if nfc_strip(t) in bench_keys)


# ---------- stage 2: train ----------

def load_student(device):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    st = SentenceTransformer(STUDENT_REPO, device=device)
    st.eval()
    return st


def encode_student(st, texts, bs=128):
    import torch
    st.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = st.tokenizer([E5_PREFIX + t for t in texts[i:i + bs]], padding=True,
                               truncation=True, max_length=MAX_LEN, return_tensors="pt").to(st.device)
            out.append(st(enc)["sentence_embedding"].float().cpu().numpy())
    return l2norm(np.concatenate(out))


def teacher_topk(emb, k=TOPK, chunk=4096):
    """Top-k NN indices (self excluded) via chunked fp16 matmul on GPU."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.from_numpy(emb).to(dev).half()
    idx = torch.empty(emb.shape[0], k, dtype=torch.long, device="cpu")
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, emb.shape[0], chunk):
            e = min(s + chunk, emb.shape[0])
            sims = T[s:e] @ T.T
            sims[torch.arange(e - s), torch.arange(s, e)] = -2.0
            idx[s:e] = torch.topk(sims, k, dim=1, largest=True, sorted=True).indices.cpu()
            if (s // chunk) % 2 == 0:
                print(f"  topk {e}/{emb.shape[0]}", flush=True)
    print(f"  topk done in {time.perf_counter() - t0:.1f}s", flush=True)
    return idx.numpy()


def pretokenize(st, texts):
    tok = st.tokenizer
    enc = tok([E5_PREFIX + t for t in texts], padding=False, truncation=True, max_length=MAX_LEN)
    return [{"input_ids": ids, "attention_mask": am}
            for ids, am in zip(enc["input_ids"], enc["attention_mask"])]


def kl_loss(teacher_emb_gpu, I, s_emb, dims, tau):
    import torch
    import torch.nn.functional as F
    n = len(I)
    eye = torch.eye(n, device=s_emb.device, dtype=torch.bool)
    with torch.no_grad():
        Tm = ((teacher_emb_gpu[I] @ teacher_emb_gpu.T)[:, I].float() / tau).masked_fill(eye, -1e4)
        pt_row = F.softmax(Tm, dim=1)
        pt_col = F.softmax(Tm, dim=0)
        log_pt_row = torch.log(pt_row.clamp_min(1e-12))
        log_pt_col = torch.log(pt_col.clamp_min(1e-12))
    s_emb = s_emb.float()
    total = s_emb.new_zeros(())
    for d in dims:
        sd = F.normalize(s_emb[:, :d], p=2, dim=1)
        S = ((sd @ sd.T) / tau).masked_fill(eye, -1e4)
        log_s_row = F.log_softmax(S, dim=1)
        log_s_col = F.log_softmax(S, dim=0)
        row = (pt_row * (log_pt_row - log_s_row)).sum() / n
        col = (pt_col * (log_pt_col - log_s_col)).sum() / n
        total = total + 0.5 * (row + col)
    return total / len(dims)


def epoch_batches(perm, nn_idx, g_per_batch, rng):
    starts = rng.integers(0, TOPK - WINDOW + 1, size=len(perm))
    for s in range(0, len(perm), g_per_batch):
        used, rows = set(), []
        for gi, a in enumerate(perm[s:s + g_per_batch]):
            grp = [int(a)] + nn_idx[a, starts[s + gi]:starts[s + gi] + WINDOW].tolist()
            if used & set(grp):
                continue  # drop colliding group
            rows.extend(grp)
            used.update(grp)
        yield rows


def stage_train(results, cfg):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    if not torch.cuda.is_available():
        raise RuntimeError("training needs CUDA")

    tau, bs, epochs, lr = cfg["tau"], cfg["bs"], cfg["epochs"], cfg["lr"]
    dims = list(cfg["dims"])
    cfg_key = config_key(cfg)
    bench_texts, bench_langs, gt = load_bench()

    if TEACHER_NPY.exists():
        teacher_emb = np.load(TEACHER_NPY).astype(np.float32)
    else:
        print("[train] teacher npy missing, embedding now", flush=True)
        pool_texts_tp, _, _ = build_pool(load_bench()[0])
        teacher_emb = embed_teacher(pool_texts_tp)
        np.save(TEACHER_NPY, teacher_emb)
    pool_texts, _ = ensure_train_pool(bench_texts)
    if teacher_emb.shape[0] != len(pool_texts):
        raise RuntimeError(f"teacher npy rows {teacher_emb.shape[0]} != pool texts {len(pool_texts)}; "
                           "delete stale data/ files and rerun --stage teacher")

    st = load_student("cuda")
    feats = pretokenize(st, pool_texts)
    nn_idx = teacher_topk(teacher_emb)
    teacher_gpu = torch.from_numpy(teacher_emb).to("cuda").half()
    g_per_batch = bs // (WINDOW + 1)
    n_batches = (len(pool_texts) + g_per_batch - 1) // g_per_batch
    total_steps = n_batches * epochs
    warmup = max(1, int(WARMUP_FRAC * total_steps))

    opt = torch.optim.AdamW(st.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    pairs = make_pairs(len(bench_texts))
    print(f"[train] {cfg_key}: n={len(pool_texts)} groups/bs={g_per_batch}x{WINDOW + 1} "
          f"steps/epoch~{n_batches} total={total_steps} dims={dims} tau={tau}", flush=True)

    best = {"score": -2.0, "epoch": -1, "metrics": None, "sd": None}
    per_epoch, opt_steps, t_train = [], 0, time.perf_counter()
    for epoch in range(epochs):
        rng = np.random.default_rng(SEED + epoch)
        perm = rng.permutation(len(pool_texts))
        st.train()
        ep_t0, losses, nb = time.perf_counter(), [], 0
        for rows in epoch_batches(perm, nn_idx, g_per_batch, rng):
            I = torch.tensor(rows, dtype=torch.long, device="cuda")
            batch = st.tokenizer.pad([feats[j] for j in rows], padding=True, return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_emb = st(batch)["sentence_embedding"]
            loss = kl_loss(teacher_gpu, I, s_emb, dims, tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            losses.append(loss.item())
            opt_steps += 1
            nb += 1
            if nb % 500 == 0:
                print(f"  ep{epoch} step {nb}/{n_batches} loss={np.mean(losses[-500:]):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
        emb_b = encode_student(st, bench_texts)
        q = quality_report(gt, emb_b, bench_langs, pairs)
        secs = time.perf_counter() - ep_t0
        row = {"epoch": epoch, "train_loss": r(np.mean(losses)), "secs": round(secs, 1), **q}
        per_epoch.append(row)
        print(f"  ep{epoch} loss={row['train_loss']} sp={q['pairwise_spearman_20k_pairs']} "
              f"jac={q['knn_jaccard_at10']} mrr={q['mrr_at10_of_gt_top1']} "
              f"top1={q['top1_agreement']} top1xl={q['top1_agreement_crosslingual']} "
              f"ari={q['ari_50_clusters']} {secs:.0f}s", flush=True)
        if q["pairwise_spearman_20k_pairs"] > best["score"]:
            best = {"score": q["pairwise_spearman_20k_pairs"], "epoch": epoch, "metrics": q,
                    "sd": {k: v.detach().cpu().clone() for k, v in st.state_dict().items()}}
            best["emb_bench"] = emb_b

    train_s = time.perf_counter() - t_train
    print(f"[train] best epoch {best['epoch']} (spearman {best['score']}); "
          f"total {train_s / 60:.1f} min", flush=True)

    st.load_state_dict(best["sd"])
    STUDENT_DIR.mkdir(parents=True, exist_ok=True)
    st.save_pretrained(str(STUDENT_DIR), create_model_card=False)
    with open(STUDENT_DIR / "train_meta.json", "w") as f:
        json.dump({"config": cfg, "config_key": cfg_key, "best_epoch": best["epoch"],
                   "best_metrics": best["metrics"], "student_repo": STUDENT_REPO,
                   "teacher_repo": TEACHER_REPO, "pool": {"n": len(pool_texts)}, "max_len": MAX_LEN},
                  f, ensure_ascii=False, indent=2)
    np.save(STUDENT_TORCH_EMB, best["emb_bench"].astype(np.float32))

    entry = results["train"].setdefault(cfg_key, {})
    entry.update({"config": cfg, "student_dir": str(STUDENT_DIR), "per_epoch": per_epoch,
                  "best_epoch": best["epoch"], "final_metrics": best["metrics"],
                  "train_s": round(train_s, 1), "steps": opt_steps,
                  "student_repo": STUDENT_REPO, "teacher_repo": TEACHER_REPO})
    del st, best["sd"]
    gc.collect()
    torch.cuda.empty_cache()


def config_key(cfg):
    key = f"tau{cfg['tau']:g}_bs{cfg['bs']}_ep{cfg['epochs']}"
    if list(cfg["dims"]) != DEFAULT_DIMS:
        key += "_d" + "-".join(str(d) for d in cfg["dims"])
    return key


# ---------- stage 3: onnx ----------

def export_onnx(st, out_path):
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
    wrap = OnnxWrap(st[0].auto_model).eval().to("cpu")
    tti = torch.zeros_like(probe["input_ids"])
    with torch.no_grad():
        torch.onnx.export(
            wrap, (probe["input_ids"], probe["attention_mask"], tti), str(out_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "attention_mask": {0: "batch", 1: "seq"},
                          "token_type_ids": {0: "batch", 1: "seq"},
                          "last_hidden_state": {0: "batch", 1: "seq"}},
            opset_version=17, dynamo=False, do_constant_folding=True)


def quantize_to_int8(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for stale in MODELS_OUT.glob(int8_path.name + "*"):
        stale.unlink()  # quantizer appends to existing external data
    try:
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        external = False
    except Exception:
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                         use_external_data_format=True)
        external = True
    return sum(f.stat().st_size for f in MODELS_OUT.glob(int8_path.name + "*")), external


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


def bench_session(path, tok, texts, threads=CPU_THREADS, n_bs1=N_BS1_TEXTS):
    import onnxruntime as ort
    gc.collect()
    pre = rss_mb()
    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    peak = max(rss_mb(), pre)
    onnx_embed(sess, tok, texts[:16], bs=16)  # warmup
    peak = max(peak, rss_mb())

    lat = []
    t0 = time.perf_counter()
    for t in texts[:n_bs1]:
        t1 = time.perf_counter()
        onnx_embed(sess, tok, [t], bs=1)
        lat.append((time.perf_counter() - t1) * 1000)
    bs1_dt = time.perf_counter() - t0
    peak = max(peak, rss_mb())

    r_ = {"path": str(path), "bytes": path.stat().st_size, "load_s": round(load_s, 2),
          "session_inputs": [i.name for i in sess.get_inputs()],
          "bs1": {"texts_s": round(len(lat) / bs1_dt, 1),
                  "latency_ms": {"p50": r(np.median(lat), 2), "p95": r(np.percentile(lat, 95), 2)}}}
    for bsz in (8, 32):
        t0 = time.perf_counter()
        onnx_embed(sess, tok, texts, bs=bsz)
        dt = time.perf_counter() - t0
        peak = max(peak, rss_mb())
        r_[f"bs{bsz}"] = {"texts_s": round(len(texts) / dt, 1),
                          "ms_per_batch": round(dt / (len(texts) / bsz) * 1000, 1)}
        print(f"  {path.name} bs{bsz}: {len(texts) / dt:.1f} texts/s", flush=True)
    r_["peak_rss_mb"] = round(peak)
    r_["rss_delta_mb"] = round(peak - pre)
    del sess
    gc.collect()
    return r_


def torch_ref_embeddings(bench_texts):
    import torch
    from sentence_transformers import SentenceTransformer
    if STUDENT_TORCH_EMB.exists():
        emb = np.load(STUDENT_TORCH_EMB).astype(np.float32)
        if emb.shape == (len(bench_texts), 384):
            return emb, False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(CPU_THREADS if device == "cpu" else torch.get_num_threads())
    st = SentenceTransformer(str(STUDENT_DIR), device=device)
    emb = encode_student(st, bench_texts)
    del st
    if device == "cuda":
        torch.cuda.empty_cache()
    np.save(STUDENT_TORCH_EMB, emb.astype(np.float32))
    return emb, True


def stage_onnx(results, cfg):
    import torch
    from sentence_transformers import SentenceTransformer
    if not STUDENT_DIR.exists():
        raise RuntimeError(f"{STUDENT_DIR} missing; run --stage train first")

    bench_texts, bench_langs, gt = load_bench()
    pairs = make_pairs(len(bench_texts))
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    cfg_key = config_key(cfg)
    if (STUDENT_DIR / "train_meta.json").exists():
        meta = json.loads((STUDENT_DIR / "train_meta.json").read_text())
        cfg_key = meta.get("config_key", cfg_key)
    entry = results["onnx"].setdefault(cfg_key, {"model_dir": str(STUDENT_DIR)})

    t0 = time.perf_counter()
    st = SentenceTransformer(str(STUDENT_DIR), device="cuda" if torch.cuda.is_available() else "cpu")
    st.eval()
    tok = st.tokenizer
    n_params = int(sum(p.numel() for p in st.parameters()))
    export_onnx(st, ONNX_FP32)
    print(f"[onnx] fp32 exported -> {ONNX_FP32.name} ({fmt_bytes(ONNX_FP32.stat().st_size)}) "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)
    del st
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    int8_bytes, external = quantize_to_int8(ONNX_FP32, ONNX_INT8)
    print(f"[onnx] int8 quantized -> {ONNX_INT8.name} ({fmt_bytes(int8_bytes)}) "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)

    torch_ref, recomputed = torch_ref_embeddings(bench_texts)
    q_torch = quality_report(gt, torch_ref, bench_langs, pairs)
    entry["torch_fp32"] = {"n_params": n_params, "quality": q_torch,
                           "emb_cache_recomputed": recomputed}
    print(f"[onnx] torch fp32 quality: sp={q_torch['pairwise_spearman_20k_pairs']} "
          f"jac={q_torch['knn_jaccard_at10']}", flush=True)

    import onnxruntime as ort
    for tag, path in (("fp32_onnx", ONNX_FP32), ("int8_onnx", ONNX_INT8)):
        try:
            print(f"[onnx] bench {tag}: {path.name}", flush=True)
            perf = bench_session(path, tok, bench_texts)
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            emb = onnx_embed(sess, tok, bench_texts)
            del sess
            q = quality_report(gt, emb, bench_langs, pairs)
            cos = np.sum(emb * torch_ref, axis=1)
            entry[tag] = {**perf,
                          "quality_vs_gt": q,
                          "cos_vs_torch_fp32": {"mean": r(cos.mean(), 6), "min": r(cos.min(), 6)},
                          "size_mb": round(path.stat().st_size / 1024**2, 1),
                          "external_data": external if tag == "int8_onnx" else False}
            print(f"  {tag}: sp={q['pairwise_spearman_20k_pairs']} jac={q['knn_jaccard_at10']} "
                  f"cos_torch={cos.mean():.4f} p50={perf['bs1']['latency_ms']['p50']}ms "
                  f"bs8={perf['bs8']['texts_s']} t/s", flush=True)
        except Exception:
            entry[tag] = {"error": traceback.format_exc()[-3000:]}
            print(f"  ERROR ({tag}): {traceback.format_exc(limit=3)}", flush=True)


# ---------- summary ----------

def print_summary(results, cfg_key):
    tr = results["train"].get(cfg_key, {})
    onx = results["onnx"].get(cfg_key, {})
    best = tr.get("final_metrics") or {}
    t8 = onx.get("int8_onnx") or {}
    q8 = t8.get("quality_vs_gt") or {}
    q32 = (onx.get("torch_fp32") or {}).get("quality") or {}
    ref = STOCK_REF
    rows = [
        ("torch sp (20k pairs)", ref["torch_fp32"]["pairwise_spearman_20k_pairs"],
         q32.get("pairwise_spearman_20k_pairs")),
        ("torch jac@10", ref["torch_fp32"]["knn_jaccard_at10"], q32.get("knn_jaccard_at10")),
        ("torch mrr@10", ref["torch_fp32"]["mrr_at10_of_gt_top1"], q32.get("mrr_at10_of_gt_top1")),
        ("torch top1", ref["torch_fp32"]["top1_agreement"], q32.get("top1_agreement")),
        ("torch top1_xl", ref["torch_fp32"]["top1_agreement_crosslingual"],
         q32.get("top1_agreement_crosslingual")),
        ("torch ari(50)", ref["torch_fp32"]["ari_50_clusters"], q32.get("ari_50_clusters")),
        ("int8 onnx size MB", ref["onnx_int8"]["size_mb"], t8.get("size_mb")),
        ("int8 bs8 t/s", ref["onnx_int8"]["bs8_texts_s"], (t8.get("bs8") or {}).get("texts_s")),
        ("int8 p50 ms", ref["onnx_int8"]["p50_ms"], (t8.get("bs1") or {}).get("latency_ms", {}).get("p50")),
        ("int8 cos vs torch", ref["onnx_int8"]["cos_vs_torch"],
         (t8.get("cos_vs_torch_fp32") or {}).get("mean")),
        ("int8 jac@10 vs GT", ref["onnx_int8"]["knn_jaccard_at10_vs_gt"],
         q8.get("knn_jaccard_at10")),
        ("int8 sp vs GT", None, q8.get("pairwise_spearman_20k_pairs")),
    ]
    print("\n== stock e5-small (edge_compare §6 ref) vs distilled nav-e5s ==")
    print(f"{'metric':<22}{'stock-e5-small':>16}{'nav-e5s-distill':>18}")
    for name, a, b in rows:
        fa = f"{a:.3f}" if isinstance(a, float) else ("-" if a is None else str(a))
        fb = f"{b:.3f}" if isinstance(b, float) else ("-" if b is None else str(b))
        print(f"{name:<22}{fa:>16}{fb:>18}")
    if tr:
        print(f"best epoch: {tr.get('best_epoch')} of {tr.get('config', {}).get('epochs')}, "
              f"train {tr.get('train_s', 0) / 60:.1f} min")


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
    for k in ("meta", "teacher", "train", "onnx", "errors"):
        res.setdefault(k, {})
    return res


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    ap = argparse.ArgumentParser(description="Distill bge-m3 -> multilingual-e5-small for nav instructions, export ONNX int8")
    ap.add_argument("--stage", choices=["teacher", "train", "onnx", "all"], default="all")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--bs", type=int, default=DEFAULT_BS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--dims", type=str, default=",".join(map(str, DEFAULT_DIMS)),
                    help="comma-separated Matryoshka dims, e.g. 64,128,256,384")
    args = ap.parse_args()

    import torch
    import sentence_transformers
    import transformers
    cfg = {"tau": args.tau, "bs": args.bs, "epochs": args.epochs, "lr": args.lr,
           "dims": [int(d) for d in args.dims.split(",") if d.strip()],
           "max_len": MAX_LEN, "topk": TOPK, "window": WINDOW, "seed": SEED,
           "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC, "grad_clip": GRAD_CLIP,
           "teacher_batch": TEACHER_BATCH, "loss": "matryoshka_sym_kl",
           "student": STUDENT_REPO, "teacher": TEACHER_REPO}
    cfg_key = config_key(cfg)

    results = load_results()
    results["meta"].update({
        "script": "train_distill.py", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED, "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hf_hub_offline": True,
    })

    if args.bs % (WINDOW + 1) != 0:
        raise SystemExit(f"--bs must be a multiple of {WINDOW + 1} (group = 1 anchor + {WINDOW} NNs)")

    stages = ["teacher", "train", "onnx"] if args.stage == "all" else [args.stage]
    for stage in stages:
        t0 = time.perf_counter()
        try:
            if stage == "teacher":
                stage_teacher(results)
            elif stage == "train":
                stage_train(results, cfg)
            elif stage == "onnx":
                stage_onnx(results, cfg)
            results["errors"].pop(stage, None)
        except Exception:
            results["errors"][stage] = traceback.format_exc()[-3000:]
            print(f"[{stage}] ERROR: {traceback.format_exc(limit=3)}", flush=True)
        results["meta"][f"{stage}_wall_s"] = round(time.perf_counter() - t0, 1)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[{stage}] saved -> {RESULTS_PATH}", flush=True)

    print_summary(results, pick_summary_key(results, cfg_key))


def pick_summary_key(results, cfg_key):
    """Prefer the CLI config key; fall back to whatever train/onnx actually stored."""
    if cfg_key in results["train"] or cfg_key in results["onnx"]:
        return cfg_key
    for section in ("onnx", "train"):
        if results[section]:
            return list(results[section])[-1]
    return cfg_key


if __name__ == "__main__":
    main()
