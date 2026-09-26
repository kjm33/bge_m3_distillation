"""Compare small multilingual embedding models for an edge device (car navigation:
CPU-only SoC, limited RAM/storage) against the BGE-M3 baseline.

Stages (runnable separately, results merged into data/edge_compare_results.json):
  A. bench set      - 512 stratified real instructions (5 langs, JSON artifacts filtered)
  B. ground truth   - BGE-M3 fp32 CUDA, CLS pooling (replicates profile_local.py)
  C. candidates     - 7 smaller models embedded on CUDA fp32 (e5 gets "query: " prefix)
  D. quality        - Spearman(20k pairs), kNN Jaccard@10, MRR@10, ARI(50 clusters),
                      top-1 agreement (incl. cross-lingual subset)
  E. MRL sweep      - prefix truncation + re-normalization for MRL-trained models
                      (qwen3, nomic) vs naive-truncation controls (e5_small, bge_m3)
  F. CPU profile    - fresh CPU pass, 16 torch threads, fp32: load time, throughput
                      bs=1/8/32, single-text latency p50/p95, peak RSS (psutil)
  G. ONNX int8      - quantize shipped onnx/model.onnx for minilm/e5/labse/e5_base,
                      measure size, session init, throughput, latency, quality

Run from bge_m3/:  .venv/bin/python edge_compare.py --stage quality|cpu|onnx|all
Network is only touched when cache is incomplete (minilm/e5_* transformer files,
onnx/model.onnx in step G); offline-first, flip back only on failure.
"""
import argparse
import gc
import json
import os
import random
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # force cache use; allow_network() flips back

import numpy as np
import psutil
from dotenv import load_dotenv
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODELS_OUT = HERE / "models" / "edge"
SRC = Path("/home/kamil/projects/here/instruction_extraction/data/instructions_v3.jsonl")
GT_PATH = DATA / "edge_gt_fp32.npy"
RESULTS_PATH = DATA / "edge_compare_results.json"

SEED = 42
N_BENCH = 512
N_PAIRS = 20_000
K = 10
N_CLUSTERS = 50
CPU_THREADS = 16
LANG_ORDER = ["en", "de", "fr", "pl", "es"]

MODELS = OrderedDict(
    bge_m3=dict(repo="BAAI/bge-m3", prefix="", license="mit", baseline=True),
    qwen3_06b=dict(repo="Qwen/Qwen3-Embedding-0.6B", prefix="", license="apache-2.0",
                   mrl=[64, 128, 256, 512, 1024], trained_mrl=True),
    nomic_v2=dict(repo="nomic-ai/nomic-embed-text-v2-moe", prefix="", license="apache-2.0",
                  mrl=[64, 128, 256, 512, 768], trained_mrl=True,
                  skip_reason="nomic custom code (NomicBertModel) incompatible with transformers 5.17 "
                              "('get_extended_attention_mask' removed); would need transformers<5"),
    e5_base=dict(repo="intfloat/multilingual-e5-base", prefix="query: ", license="mit", onnx=True),
    labse=dict(repo="sentence-transformers/LaBSE", prefix="", license="apache-2.0", onnx=True),
    e5_small=dict(repo="intfloat/multilingual-e5-small", prefix="query: ", license="mit",
                  mrl=[96, 192, 384], trained_mrl=False, onnx=True),
    minilm=dict(repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                prefix="", license="apache-2.0", onnx=True),
    potion=dict(repo="minishlab/potion-multilingual-128M", prefix="", license="mit"),
)
CANDIDATES = [k for k, v in MODELS.items() if not v.get("baseline") and not v.get("skip_reason")]
CPU_KEYS = [k for k in ["qwen3_06b", "nomic_v2", "e5_base", "labse", "e5_small", "minilm", "potion", "bge_m3"]
            if "skip_reason" not in MODELS[k]]
ONNX_KEYS = ["minilm", "e5_small", "e5_base", "labse"]
ONNX_SKIPPED = {"qwen3_06b": "no official onnx export shipped in repo (MRL dims must stay selectable anyway)",
                "nomic_v2": "no official onnx export shipped in repo (custom code, MoE)",
                "bge_m3": "no official onnx export usable for edge comparison (2.1 GB, already quantized in quantize_compare.py)"}

RSS_NOISE_NOTE = ("RSS deltas after the first model in the CPU loop are noisy: python/torch "
                  "allocators keep freed pages, so peak_rss_mb (absolute) per stage is the more "
                  "meaningful number; deltas are upper bounds of the true model footprint.")

PROCESS = psutil.Process(os.getpid())
_network_fallbacks: list[str] = []


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


# ---------- offline/network handling ----------

@contextmanager
def allow_network():
    """Temporarily disable HF offline mode (env vars + already-imported constants)."""
    import huggingface_hub.constants as hc
    saved_env = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    saved_const = hc.HF_HUB_OFFLINE
    hc.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        hc.HF_HUB_OFFLINE = saved_const
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v


def load_with_fallback(fn, what):
    """Run an offline-first loader; on failure retry once with network allowed."""
    try:
        return fn()
    except Exception as first_err:
        print(f"  [cache miss for {what} ({type(first_err).__name__}) -> fetching needed files]")
        _network_fallbacks.append(what)
        with allow_network():
            return fn()


SNAPSHOT_PATTERNS = [
    "config.json", "tokenizer*", "vocab*", "special_tokens_map.json", "spm.model",
    "*.safetensors", "onnx/model.onnx*",
    "1_Pooling/*", "2_Normalize/*", "modules.json", "sentence_bert_config.json",
    "config_sentence_transformers.json",
]


def snapshot_dir(repo):
    from huggingface_hub import snapshot_download
    p = load_with_fallback(
        lambda: snapshot_download(repo, local_files_only=True, allow_patterns=SNAPSHOT_PATTERNS),
        f"{repo} snapshot")
    return Path(p)


def license_of(key, repo):
    fallback = MODELS[key]["license"]
    try:
        from huggingface_hub import HfApi
        info = load_with_fallback(lambda: HfApi().model_info(repo, files_metadata=False), f"{repo} license")
        lic = getattr(info.card_data, "license", None) if info.card_data else None
        return (lic or fallback), "hf_card"
    except Exception:
        return fallback, "fallback_table"


def weight_size(snap):
    """fp32 weight bytes: prefer *.safetensors, fall back to *.bin (bge-m3 main snapshot is bin-only)."""
    for pattern in ("*.safetensors", "*.bin"):
        files = sorted(p for p in snap.rglob(pattern) if p.is_file() and "openvino" not in str(p))
        if files:
            return sum(f.stat().st_size for f in files), [f.name for f in files], pattern
    return 0, [], "none"


# ---------- step A: bench set ----------

def build_bench():
    pools: dict[str, list[str]] = {}
    n_total = n_filtered = 0
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text")
            n_total += 1
            if not isinstance(text, str):
                continue
            if text.strip().startswith("{"):  # JSON completion artifacts
                n_filtered += 1
                continue
            pools.setdefault(d.get("language", "?"), []).append(text)

    rng = np.random.default_rng(SEED)
    quota = N_BENCH // len(LANG_ORDER)
    extra = N_BENCH - quota * len(LANG_ORDER)
    texts, langs, per_lang = [], [], {}
    for i, lang in enumerate(LANG_ORDER):
        pool = pools.get(lang, [])
        take = quota + (1 if i < extra else 0)
        idx = rng.permutation(len(pool))[:take]
        for j in idx:
            texts.append(pool[int(j)])
            langs.append(lang)
        per_lang[lang] = int(len(idx))

    DATA.mkdir(exist_ok=True)
    with open(DATA / "edge_bench.jsonl", "w") as f:
        for t, l in zip(texts, langs):
            f.write(json.dumps({"text": t, "language": l}, ensure_ascii=False) + "\n")

    meta = {"source": str(SRC), "n_source_records": n_total,
            "n_filtered_json_artifacts": n_filtered, "n_bench": len(texts),
            "per_language": per_lang, "seed": SEED, "path": str(DATA / "edge_bench.jsonl")}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS["bge_m3"]["repo"])
        n_tok = sum(len(x) for x in tok(texts, padding=False, truncation=False)["input_ids"])
        meta["avg_tokens_per_text"] = round(n_tok / len(texts), 1)
        meta["bench_tokens_bge_tokenizer"] = n_tok
    except Exception as e:
        meta["avg_tokens_per_text"] = f"unavailable ({type(e).__name__})"
    return texts, langs, meta


# ---------- step B: ground truth (replicates profile_local.py CLS fp32 CUDA) ----------

def build_gt(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    repo = MODELS["bge_m3"]["repo"]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to("cuda").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=8192,
                      return_tensors="pt").to("cuda")
            out.append(model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return l2norm(np.concatenate(out).astype(np.float32))


def ensure_gt(texts):
    if GT_PATH.exists():
        gt = np.load(GT_PATH).astype(np.float32)
        if gt.shape != (len(texts), 1024):
            gt = build_gt(texts)
            np.save(GT_PATH, gt)
        return gt, {"reused_existing_file": True}
    if not torch.cuda_is_available() if False else False:
        pass
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"{GT_PATH} missing and no CUDA available to build it")
    gt = build_gt(texts)
    np.save(GT_PATH, gt)
    return gt, {"reused_existing_file": False, "pooling": "cls", "device": "cuda", "dtype": "fp32"}


# ---------- model loading ----------

def load_st(key, device):
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = MODELS[key]
    def _load():
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        return SentenceTransformer(cfg["repo"], device=device, trust_remote_code=True)
    st = load_with_fallback(_load, f"{cfg['repo']} (sentence-transformers)")
    st.eval()
    return st


def encode_st(st, texts, prefix="", bs=64):
    if prefix:
        texts = [prefix + t for t in texts]
    arr = st.encode(texts, batch_size=bs, convert_to_numpy=True, show_progress_bar=False)
    return l2norm(np.asarray(arr, dtype=np.float32))


def load_potion():
    from model2vec import StaticModel
    return load_with_fallback(lambda: StaticModel.from_pretrained(MODELS["potion"]["repo"]),
                              f"{MODELS['potion']['repo']} (model2vec)")


# ---------- step D: quality metrics (all embeddings L2-normalized) ----------

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
    """Jaccard@10, MRR@10 (rank of GT top-1 in cand top-10), top-1 agreement + cross-lingual subset."""
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
    out = {"knn_jaccard_at10": r(np.mean(jacs)),
           "mrr_at10_of_gt_top1": r(np.mean(rrs)),
           "top1_agreement": r(top1_hits / n),
           "top1_agreement_crosslingual": r(xl_hits / xl_rows) if xl_rows else None,
           "n_crosslingual_top1_texts": xl_rows}
    return out


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


def truncation_metrics(gt, cand_prefix, pairs):
    """Spearman + Jaccard@10 of a truncated (re-normalized) embedding vs the FULL ground truth."""
    t = l2norm(cand_prefix[:, :].astype(np.float32) if False else cand_prefix)
    out = {"pairwise_spearman_20k_pairs": similarity_spearman(gt, t, pairs)}
    S_gt, S_t = gt @ gt.T, t @ t.T
    np.fill_diagonal(S_gt, -np.inf)
    np.fill_diagonal(S_t, -np.inf)
    ord_gt = np.argsort(-S_gt, axis=1)
    ord_t = np.argsort(-S_t, axis=1)
    jacs = [len(set(ord_gt[r_, :K].tolist()) & set(ord_t[r_, :K].tolist())) /
            len(set(ord_gt[r_, :K].tolist()) | set(ord_t[r_, :K].tolist())) for r_ in range(len(gt))]
    out["knn_jaccard_at10"] = r(np.mean(jacs))
    return out


# ---------- step C+D+E: quality stage (GPU) ----------

def run_quality(results, texts, langs):
    import torch
    gt, gt_info = ensure_gt(texts)
    results["bench"] = build_bench()[2]
    results["ground_truth"] = {"model": MODELS["bge_m3"]["repo"], "n": len(gt), "dim": gt.shape[1],
                               "dtype": "fp32", "l2_normalized": True, **gt_info}
    pairs = make_pairs(len(texts))
    models = results["models"]

    bm = models.setdefault("bge_m3", {})
    bm.setdefault("meta", {"repo": MODELS["bge_m3"]["repo"], "role": "baseline/ground_truth",
                           "dim": int(gt.shape[1]), "impl": "transformers AutoModel (CLS)"})
    bm["meta"].setdefault("license", license_of("bge_m3", MODELS["bge_m3"]["repo"]))
    bm["meta"].setdefault("weights_fp32", weight_size(snapshot_dir(MODELS["bge_m3"]["repo"]))[0:2])
    bm["quality_vs_gt"] = {"pairwise_spearman_20k_pairs": 1.0, "knn_jaccard_at10": 1.0,
                           "mrr_at10_of_gt_top1": 1.0, "top1_agreement": 1.0,
                           "ari_50_clusters": None, "note": "is the ground truth"}

    for key in CANDIDATES:
        cfg = MODELS[key]
        print(f"[quality] {key} ({cfg['repo']})", flush=True)
        try:
            meta = {"repo": cfg["repo"], "prefix": cfg["prefix"] or None,
                    "license": license_of(key, cfg["repo"])}
            snap = snapshot_dir(cfg["repo"])
            wbytes, wfiles, wfmt = weight_size(snap)
            meta["weights_fp32"] = {"bytes": wbytes, "files": wfiles, "format": wfmt}
            if key == "potion":
                m = load_potion()
                emb = l2norm(m.encode(texts, batch_size=64, show_progress_bar=False).astype(np.float32))
                meta["impl"] = "model2vec.StaticModel (static embeddings)"
                meta["dim"] = int(emb.shape[1])
                meta["n_params"] = int(m.embedding.size)
                meta["n_params_note"] = "vocab x dim static lookup table (no transformer layers)"
                del m
            else:
                st = load_st(key, "cuda")
                emb = encode_st(st, texts, cfg["prefix"])
                meta["impl"] = "sentence_transformers (fp32)"
                meta["dim"] = int(emb.shape[1])
                meta["n_params"] = int(sum(p.numel() for p in st.parameters()))
                meta["max_seq_length"] = int(getattr(st, "max_seq_length", 0)) or None
                del st
                torch.cuda.empty_cache()
            np.save(DATA / f"edge_emb_{key}.npy", emb.astype(np.float32))
            q = quality_report(gt, emb, langs, pairs)
            entry = models.setdefault(key, {})
            entry["meta"], entry["quality"] = meta, q
            models[key] = entry
            results["errors"].pop(key, None)
            print(f"  dim={meta['dim']} params={meta.get('n_params')} "
                  f"spearman={q['pairwise_spearman_20k_pairs']} jac@10={q['knn_jaccard_at10']}", flush=True)

            if cfg.get("mrl"):
                sweep = {"trained_mrl": cfg.get("trained_mrl", False), "full_dim": meta["dim"], "dims": {}}
                for d in cfg["mrl"]:
                    sweep["dims"][str(d)] = truncation_metrics(gt, emb[:, :d], pairs)
                results["mrl_sweep"][key] = sweep
        except Exception:
            results["errors"][key] = {"stage": "quality", "error": traceback.format_exc()[-3000:]}
            print(f"  ERROR: {traceback.format_exc(limit=3)}", flush=True)

    # naive-truncation control on the ground truth itself
    sweep = {"trained_mrl": False, "full_dim": int(gt.shape[1]), "dims": {},
             "note": "bge_m3 is NOT MRL-trained; control for the MRL sweep above"}
    for d in [128, 256, 512, 1024]:
        sweep["dims"][str(d)] = truncation_metrics(gt, gt[:, :d], pairs)
    results["mrl_sweep"]["bge_m3_naive_truncation"] = sweep


# ---------- step F: CPU resource profile ----------

def run_cpu(results, texts):
    import torch
    torch.set_num_threads(CPU_THREADS)
    results["cpu_env"] = {"device": "cpu", "dtype": "fp32", "torch_threads": CPU_THREADS,
                          "torch_get_num_threads": torch.get_num_threads(),
                          "note": RSS_NOISE_NOTE, "n_warmup": 16, "n_bs1_texts": 128}
    models = results["models"]
    stage_peak = rss_mb()

    for key in CPU_KEYS:
        cfg = MODELS[key]
        print(f"[cpu] {key}", flush=True)
        try:
            gc.collect()
            pre = rss_mb()
            t0 = time.perf_counter()
            if key == "potion":
                m = load_potion()
                enc = lambda ts, bs: m.encode(ts, batch_size=bs, show_progress_bar=False)
                params = int(m.embedding.size)
            else:
                st = load_st(key, "cpu")
                enc = lambda ts, bs: st.encode(ts, batch_size=bs, convert_to_numpy=True,
                                               show_progress_bar=False)
                params = int(sum(p.numel() for p in st.parameters()))
            load_s = time.perf_counter() - t0
            peak = rss_mb()

            enc(texts[:16], 16)  # warmup
            peak = max(peak, rss_mb())

            # bs=1: 128 sequential single-text requests -> throughput + latency distribution
            lat = []
            t0 = time.perf_counter()
            for t in texts[:128]:
                t1 = time.perf_counter()
                enc([t], 1)
                lat.append((time.perf_counter() - t1) * 1000)
            bs1_dt = time.perf_counter() - t0
            peak = max(peak, rss_mb())

            r_ = {"impl": "model2vec.StaticModel" if key == "potion" else "sentence_transformers",
                  "n_params": params, "load_s": round(load_s, 2),
                  "bs1": {"texts_s": round(len(lat) / bs1_dt, 1),
                          "latency_ms": {"p50": r(np.median(lat), 1), "p95": r(np.percentile(lat, 95), 1)}}}
            for bs in (8, 32):
                t0 = time.perf_counter()
                enc(texts, bs)
                dt = time.perf_counter() - t0
                peak = max(peak, rss_mb())
                r_[f"bs{bs}"] = {"texts_s": round(len(texts) / dt, 1),
                                 "ms_per_batch": round(dt / (len(texts) / bs) * 1000, 1)}
                print(f"  bs{bs}: {len(texts)/dt:.1f} texts/s", flush=True)

            r_["peak_rss_mb"] = round(peak)
            r_["rss_delta_mb"] = round(peak - pre)
            stage_peak = max(stage_peak, peak)
            entry = models.setdefault(key, {})
            entry["meta"] = {**entry.get("meta", {}), "repo": cfg["repo"]}
            entry["cpu"] = r_
            results["errors"].pop(key, None)
            if key != "potion":
                del st
            else:
                del m
            gc.collect()
        except Exception:
            results["errors"][key] = {"stage": "cpu", "error": traceback.format_exc()[-3000:]}
            print(f"  ERROR: {traceback.format_exc(limit=3)}", flush=True)
    results["cpu_env"]["stage_peak_rss_mb"] = round(stage_peak)


# ---------- step G: ONNX int8 for small edge candidates ----------

def get_onnx_fp32(repo):
    from huggingface_hub import hf_hub_download
    fname = "onnx/model.onnx"
    return Path(load_with_fallback(lambda: hf_hub_download(repo, fname, local_files_only=True),
                                   f"{repo} onnx/model.onnx"))


def pooling_mode(snap):
    p = snap / "1_Pooling" / "config.json"
    if p.exists():
        c = json.loads(p.read_text())
        if c.get("pooling_mode_cls_token"):
            return "cls"
        if c.get("pooling_mode_mean_tokens"):
            return "mean"
        if c.get("pooling_mode_lasttoken"):
            return "lasttoken"
    return "mean"


def max_seq_len(snap):
    p = snap / "sentence_bert_config.json"
    if p.exists():
        c = json.loads(p.read_text())
        return int(c.get("max_seq_length", 512))
    return 512


def dense_from_snapshot(snap):
    """LaBSE ships a 2_Dense (linear 768->768, Tanh) after pooling; replicate it exactly."""
    cfgp = snap / "2_Dense" / "config.json"
    if not cfgp.exists():
        return None
    cfg = json.loads(cfgp.read_text())
    wp = snap / "2_Dense" / "model.safetensors"
    if not wp.exists():
        return None
    from safetensors.torch import load_file
    sd = load_file(str(wp))
    W = sd["linear.weight"].float().numpy()
    b = sd["linear.bias"].float().numpy() if "linear.bias" in sd else None
    act = "tanh" if "Tanh" in str(cfg.get("activation_function", "")) else "linear"
    return {"W": W, "b": b, "act": act, "config": cfg}


def onnx_embed(sess, tok, ts, pool, dense, max_len, bs=8, prefix=""):
    if prefix:
        ts = [prefix + t for t in ts]
    outs = []
    for i in range(0, len(ts), bs):
        chunk = ts[i:i + bs]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="np")
        feed = {inp.name: enc[inp.name].astype(np.int64) for inp in sess.get_inputs() if inp.name in enc}
        if "token_type_ids" in [i.name for i in sess.get_inputs()] and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
        h = next(o for o in sess.run(None, feed) if o.ndim == 3)  # token-level output
        mask = enc["attention_mask"].astype(np.float32)
        if pool == "cls":
            v = h[:, 0, :]
        elif pool == "lasttoken":
            idx = np.maximum(mask.sum(1).astype(int) - 1, 0)
            v = h[np.arange(len(chunk)), idx]
        else:  # masked mean
            m_ = mask[:, :, None]
            v = (h * m_).sum(1) / np.maximum(m_.sum(1), 1.0)
        if dense is not None:
            v = v @ dense["W"].T
            if dense["b"] is not None:
                v = v + dense["b"]
            if dense["act"] == "tanh":
                v = np.tanh(v)
        outs.append(v.astype(np.float32))
    return l2norm(np.concatenate(outs))


def quantize_to_int8(fp32_path, int8_path):
    import onnxruntime as ort  # noqa: F401  (import guards env sanity)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for stale in MODELS_OUT.glob(int8_path.name + "*"):
        stale.unlink()  # quantizer appends to existing external data
    try:
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        external = False
    except Exception:  # proto > 2GB edge case
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                         use_external_data_format=True)
        external = True
    files = list(MODELS_OUT.glob(int8_path.name + "*"))
    return sum(f.stat().st_size for f in files), external


def torch_ref_embeddings(key, texts, prefix):
    """fp32 PyTorch reference from step C; recompute on the fly if the npy is missing."""
    path = DATA / f"edge_emb_{key}.npy"
    if path.exists():
        return np.load(path).astype(np.float32), False
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if key == "potion":
        m = load_potion()
        emb = l2norm(m.encode(texts, batch_size=64, show_progress_bar=False).astype(np.float32))
    else:
        st = load_st(key, device)
        emb = encode_st(st, texts, prefix, bs=32 if device == "cpu" else 64)
        del st
        if device == "cuda":
            torch.cuda.empty_cache()
    np.save(path, emb.astype(np.float32))
    return emb, True


def run_onnx(results, texts, langs):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    gt, _ = ensure_gt(texts)
    pairs = make_pairs(len(texts))
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    models = results["models"]

    for key in ONNX_SKIPPED:
        entry = models.setdefault(key, {"meta": {"repo": MODELS[key]["repo"]}})
        entry["onnx_int8"] = {"skipped": ONNX_SKIPPED[key]}

    for key in ONNX_KEYS:
        cfg = MODELS[key]
        print(f"[onnx] {key} ({cfg['repo']})", flush=True)
        try:
            snap = snapshot_dir(cfg["repo"])
            fp32 = get_onnx_fp32(cfg["repo"])
            fp32_bytes = fp32.stat().st_size
            data_file = fp32.parent / (fp32.name + "_data")
            if data_file.exists():
                fp32_bytes += data_file.stat().st_size

            tok = load_with_fallback(lambda: AutoTokenizer.from_pretrained(cfg["repo"]),
                                     f"{cfg['repo']} tokenizer")
            pool = pooling_mode(snap)
            dense = dense_from_snapshot(snap)
            max_len = max_seq_len(snap)

            t0 = time.perf_counter()
            int8_path = MODELS_OUT / f"{key}_int8.onnx"
            int8_bytes, external = quantize_to_int8(fp32, int8_path)
            quant_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
            init_s = time.perf_counter() - t0
            onnx_embed(sess, tok, texts[:16], pool, dense, max_len, prefix=cfg["prefix"])  # warmup

            t0 = time.perf_counter()
            emb8 = onnx_embed(sess, tok, texts, pool, dense, max_len, prefix=cfg["prefix"])
            bs8_dt = time.perf_counter() - t0

            lat = []
            for t in texts[:20]:
                t1 = time.perf_counter()
                onnx_embed(sess, tok, [t], pool, dense, max_len, prefix=cfg["prefix"])
                lat.append((time.perf_counter() - t1) * 1000)
            input_names = [i.name for i in sess.get_inputs()]
            del sess

            # quality: int8 vs fp32 PyTorch embeddings (same pipeline), plus fp32-onnx sanity
            torch_ref, recomputed = torch_ref_embeddings(key, texts, cfg["prefix"])
            cos = np.sum(emb8 * torch_ref, axis=1)
            sess32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
            emb32 = onnx_embed(sess32, tok, texts, pool, dense, max_len, prefix=cfg["prefix"])
            cos32 = np.sum(emb32 * torch_ref, axis=1)
            del sess32
            S_gt, S_8 = gt @ gt.T, emb8 @ emb8.T
            np.fill_diagonal(S_gt, -np.inf)
            np.fill_diagonal(S_8, -np.inf)
            jacs = [len(set(np.argsort(-S_gt[i])[:K]) & set(np.argsort(-S_8[i])[:K])) / (2 * K - len(set(np.argsort(-S_gt[i])[:K]) & set(np.argsort(-S_8[i])[:K]))) for i in range(len(gt))]

            r_ = {"fp32_onnx_bytes": fp32_bytes, "int8_onnx_bytes": int8_bytes,
                  "compression_ratio": round(fp32_bytes / int8_bytes, 2),
                  "quantization": {"method": "quantize_dynamic", "weight_type": "QInt8",
                                   "use_external_data_format": external, "wall_s": round(quant_s, 1)},
                  "session_init_s": round(init_s, 2), "session_inputs": input_names,
                  "pooling": pool, "dense_applied": dense is not None,
                  "max_seq_length": max_len, "prefix": cfg["prefix"] or None,
                  "bs8": {"texts_s": round(len(texts) / bs8_dt, 1),
                          "ms_per_batch": round(bs8_dt / (len(texts) / 8) * 1000, 1)},
                  "latency_ms": {"p50": r(np.median(lat), 1), "p95": r(np.percentile(lat, 95), 1)},
                  "quality": {"int8_vs_torch_fp32": {"cosine_mean": r(cos.mean(), 6), "cosine_min": r(cos.min(), 6)},
                              "fp32_onnx_vs_torch_fp32": {"cosine_mean": r(cos32.mean(), 6), "cosine_min": r(cos32.min(), 6)},
                              "knn_jaccard_at10_vs_gt": r(np.mean(jacs)),
                              "torch_ref_recomputed_in_stage": recomputed}}
            entry = models.setdefault(key, {"meta": {"repo": cfg["repo"]}})
            entry["onnx_int8"] = r_
            results["errors"].pop(key, None)
            print(f"  int8 {fmt_bytes(int8_bytes)} (fp32 {fmt_bytes(fp32_bytes)}) "
                  f"bs8 {r_['bs8']['texts_s']} texts/s cos_mean={r_['quality']['int8_vs_torch_fp32']['cosine_mean']}",
                  flush=True)
        except Exception:
            results["errors"][key] = {"stage": "onnx", "error": traceback.format_exc()[-3000:]}
            print(f"  ERROR: {traceback.format_exc(limit=3)}", flush=True)


# ---------- summary printing ----------

def print_summary(results):
    models = results.get("models", {})

    q_rows = [(k, m.get("meta", {}), m.get("quality", {})) for k, m in models.items() if m.get("quality")]
    if q_rows:
        print("\n== Quality vs BGE-M3 fp32 CLS ground truth (512 edge_bench texts, normalized) ==")
        print(f"{'model':<22}{'dim':>6}{'params':>10}{'weights':>10}{'spear':>8}{'jac@10':>8}"
              f"{'mrr@10':>8}{'ari':>7}{'top1':>7}{'top1xl':>8}")
        for k, meta, q in q_rows:
            print(f"{k:<22}{str(meta.get('dim', '?')):>6}"
                  f"{(str(round(meta['n_params'] / 1e6, 1)) + 'M') if meta.get('n_params') else '?':>10}"
                  f"{fmt_bytes(meta['weights_fp32']['bytes']) if meta.get('weights_fp32', {}).get('bytes') else '?':>10}"
                  f"{q.get('pairwise_spearman_20k_pairs') or 0:>8.3f}"
                  f"{q.get('knn_jaccard_at10') or 0:>8.3f}"
                  f"{q.get('mrr_at10_of_gt_top1') or 0:>8.3f}"
                  f"{(q.get('ari_50_clusters') if q.get('ari_50_clusters') is not None else 0):>7.3f}"
                  f"{q.get('top1_agreement') or 0:>7.3f}"
                  f"{(q.get('top1_agreement_crosslingual') if q.get('top1_agreement_crosslingual') is not None else 0):>8.3f}")

    cpu_rows = [(k, m.get("cpu")) for k, m in models.items() if m.get("cpu")]
    if cpu_rows:
        print(f"\n== CPU fp32 profile ({results.get('cpu_env', {}).get('torch_threads', '?')} torch threads) ==")
        print(f"{'model':<22}{'load_s':>8}{'bs1 t/s':>9}{'bs8 t/s':>9}{'bs32 t/s':>10}"
              f"{'p50 ms':>9}{'p95 ms':>9}{'peakRSS MB':>12}{'dRSS MB':>9}")
        for k, c in cpu_rows:
            print(f"{k:<22}{c['load_s']:>8.2f}{c['bs1']['texts_s']:>9.1f}{c['bs8']['texts_s']:>9.1f}"
                  f"{c['bs32']['texts_s']:>10.1f}{c['bs1']['latency_ms']['p50']:>9.1f}"
                  f"{c['bs1']['latency_ms']['p95']:>9.1f}{c['peak_rss_mb']:>12}{c['rss_delta_mb']:>9}")

    mrl = results.get("mrl_sweep", {})
    if mrl:
        print("\n== Matryoshka / truncation sweep (prefix dims, re-normalized, vs full GT) ==")
        print(f"{'model':<26}{'MRL':>5}{'dim':>6}{'spear':>8}{'jac@10':>8}")
        for k, s in mrl.items():
            for d, v in s["dims"].items():
                print(f"{k:<26}{('yes' if s['trained_mrl'] else 'no'):>5}{d:>6}"
                      f"{v['pairwise_spearman_20k_pairs']:>8.3f}{v['knn_jaccard_at10']:>8.3f}")

    onnx_rows = [(k, m.get("onnx_int8")) for k, m in models.items()
                 if isinstance(m.get("onnx_int8"), dict) and "skipped" not in m["onnx_int8"]]
    if onnx_rows:
        print("\n== ONNX int8 (dynamic QInt8, CPUExecutionProvider) ==")
        print(f"{'model':<22}{'fp32':>9}{'int8':>9}{'ratio':>7}{'init_s':>8}{'bs8 t/s':>9}"
              f"{'p50 ms':>9}{'cos_torch':>10}{'jac@10gt':>10}")
        for k, o in onnx_rows:
            print(f"{k:<22}{fmt_bytes(o['fp32_onnx_bytes']):>9}{fmt_bytes(o['int8_onnx_bytes']):>9}"
                  f"{o['compression_ratio']:>7}{o['session_init_s']:>8.2f}{o['bs8']['texts_s']:>9.1f}"
                  f"{o['latency_ms']['p50']:>9.1f}{o['quality']['int8_vs_torch_fp32']['cosine_mean']:>10.4f}"
                  f"{o['quality']['knn_jaccard_at10_vs_gt']:>10.3f}")
        for k, m in models.items():
            o = m.get("onnx_int8")
            if isinstance(o, dict) and "skipped" in o:
                print(f"  (onnx skipped: {k} - {o['skipped']})")

    if results.get("errors"):
        print("\n== Errors ==")
        for k, e in results["errors"].items():
            print(f"  {k} [{e['stage']}]: {e['error'].strip().splitlines()[-1] if e['error'].strip() else '?'}")


# ---------- main ----------

def load_results():
    if RESULTS_PATH.exists():
        try:
            res = json.loads(RESULTS_PATH.read_text())
            for k in ("meta", "bench", "models", "mrl_sweep", "errors"):
                res.setdefault(k, {})
            return res
        except Exception as e:
            print(f"warning: could not parse existing results ({e}); starting fresh")
    return {"meta": {}, "bench": {}, "models": {}, "mrl_sweep": {}, "errors": {}}


def main():
    load_dotenv(HERE.parents[1] / ".env")  # HF_TOKEN for any fallback downloads
    random.seed(SEED)
    np.random.seed(SEED)

    ap = argparse.ArgumentParser(description="Edge embedding model comparison vs BGE-M3")
    ap.add_argument("--stage", choices=["quality", "cpu", "onnx", "all"], default="all")
    args = ap.parse_args()

    import torch
    import sentence_transformers
    import transformers
    results = load_results()
    results["meta"].update({
        "script": "edge_compare.py",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hf_hub_offline_default": True,
        "rss_note": RSS_NOISE_NOTE,
    })
    results["meta"]["network_fallbacks"] = sorted(set(_network_fallbacks + results["meta"].get("network_fallbacks", [])))

    texts, langs, bench_meta = build_bench()
    print(f"bench: {bench_meta['n_bench']} texts {bench_meta['per_language']} "
          f"({bench_meta['n_filtered_json_artifacts']} JSON artifacts filtered, "
          f"~{bench_meta['avg_tokens_per_text']} tok/text)", flush=True)
    if args.stage in ("quality", "all"):
        results["bench"] = bench_meta

    stages = ["quality", "cpu", "onnx"] if args.stage == "all" else [args.stage]
    results["skipped"] = {k: v["skip_reason"] for k, v in MODELS.items() if v.get("skip_reason")}
    for stage in stages:
        if stage == "quality":
            run_quality(results, texts, langs)
        elif stage == "cpu":
            run_cpu(results, texts)
        elif stage == "onnx":
            run_onnx(results, texts, langs)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[{stage}] saved -> {RESULTS_PATH}", flush=True)

    print_summary(results)


if __name__ == "__main__":
    main()
