"""Quantize BGE-M3 ONNX fp32 -> dynamic int8; compare size, speed, RAM, embedding quality.

Quality reference: data/groundtruth_fp32.npy (PyTorch fp32 GPU, CLS-pooled).
Config: fp32 ONNX CPU vs int8 ONNX CPU, 512 real multilingual instruction texts.
"""
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from onnxruntime.quantization import QuantType, quantize_dynamic
from scipy.stats import spearmanr
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA = HERE / "data"
FP32 = MODELS / "model.onnx"
FP32_DATA = MODELS / "model.onnx_data"
INT8 = MODELS / "model_int8.onnx"
MODEL_ID = "BAAI/bge-m3"
SEED = 0
N_PAIRS = 20000

texts = json.loads((DATA / "bench_texts.json").read_text())
gt = np.load(DATA / "groundtruth_fp32.npy").astype(np.float32)
gt /= np.linalg.norm(gt, axis=1, keepdims=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
proc = psutil.Process(os.getpid())

enc_all = tok(texts, padding=False, truncation=False)
n_tokens = sum(len(x) for x in enc_all["input_ids"])


def rss_mb():
    return proc.memory_info().rss / 1024**2


def encode(sess, ts):
    enc = tok(ts, padding=True, truncation=True, max_length=8192, return_tensors="np")
    feed = {i.name: enc[i.name].astype(np.int64) for i in sess.get_inputs() if i.name in enc}
    outs = sess.run(None, feed)
    h = next(o for o in outs if o.ndim == 3)[:, 0, :]  # token-level output -> CLS pool
    h = h.astype(np.float32)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def quality(emb):
    cos = np.sum(emb * gt, axis=1)
    n = len(gt)
    rng = np.random.default_rng(SEED)
    i = rng.integers(0, n, N_PAIRS + 1000)
    j = rng.integers(0, n, N_PAIRS + 1000)
    keep = i != j
    i, j = i[keep][:N_PAIRS], j[keep][:N_PAIRS]
    gt_sim = np.sum(gt[i] * gt[j], axis=1)
    cfg_sim = np.sum(emb[i] * emb[j], axis=1)
    rho = spearmanr(gt_sim, cfg_sim).statistic

    S_gt, S_c = gt @ gt.T, emb @ emb.T
    np.fill_diagonal(S_gt, -np.inf)
    np.fill_diagonal(S_c, -np.inf)
    jacs, rrs = [], []
    for r in range(100):
        top_gt = set(np.argsort(-S_gt[r])[:10].tolist())
        top_c = set(np.argsort(-S_c[r])[:10].tolist())
        jacs.append(len(top_gt & top_c) / len(top_gt | top_c))
        gt_top1 = int(np.argmax(S_gt[r]))
        rank = int(np.where(np.argsort(-S_c[r]) == gt_top1)[0][0]) + 1
        rrs.append(1.0 / rank)
    return {"cosine_min": round(float(cos.min()), 6),
            "cosine_mean": round(float(cos.mean()), 6),
            "max_abs_coord_diff": round(float(np.abs(emb - gt).max()), 6),
            "pairwise_spearman": round(float(rho), 6),
            "retrieval_jaccard_at10_mean": round(float(np.mean(jacs)), 6),
            "retrieval_mrr_of_gt_top1": round(float(np.mean(rrs)), 6)}


def bench(tag, path):
    gc.collect()
    base = rss_mb()
    peak = base
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    peak = max(peak, rss_mb())
    input_names = [i.name for i in sess.get_inputs()]
    encode(sess, texts[:32])  # warmup
    peak = max(peak, rss_mb())

    r = {"config": tag, "session_inputs": input_names, "load_s": round(load_s, 2),
         "baseline_rss_mb": round(base), "peak_rss_mb": round(peak)}
    embs = None
    for bs in [1, 8, 32]:
        t0 = time.perf_counter()
        outs = [encode(sess, texts[k:k + bs]) for k in range(0, len(texts), bs)]
        dt = time.perf_counter() - t0
        peak = max(peak, rss_mb())
        if bs == 32:
            embs = np.concatenate(outs)
        r[f"bs{bs}"] = {"texts_s": round(len(texts) / dt, 1),
                        "tokens_s": round(n_tokens / dt),
                        "ms_per_batch": round(dt / (len(texts) / bs) * 1000, 1)}
        print(f"  {tag} bs{bs}: {len(texts)/dt:.1f} texts/s", flush=True)

    lat = []
    for t in texts[:20]:
        t0 = time.perf_counter()
        encode(sess, [t])
        lat.append((time.perf_counter() - t0) * 1000)
    peak = max(peak, rss_mb())
    r["latency_ms"] = {"p50": round(float(np.median(lat)), 1),
                       "p95": round(float(np.percentile(lat, 95)), 1)}
    r["peak_rss_mb"] = round(peak)
    r["rss_delta_mb"] = round(peak - base)
    del sess
    return r, embs


def quantize():
    t0 = time.perf_counter()
    for stale in MODELS.glob("model_int8.onnx*"):  # quantizer appends to existing external data
        stale.unlink()
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QUInt8,
                     use_external_data_format=True,  # fp32 proto > 2 GB, cannot serialize inline
                     extra_options={"WeightSymmetric": True})
    return time.perf_counter() - t0


results = {"model": MODEL_ID, "n_texts": len(texts), "n_tokens": n_tokens,
           "avg_tokens_per_text": round(n_tokens / len(texts), 1), "seed": SEED}

fp32_bytes = FP32.stat().st_size + (FP32_DATA.stat().st_size if FP32_DATA.exists() else 0)

print("quantizing fp32 -> dynamic int8 (QUInt8, symmetric weights)...", flush=True)
q_s = quantize()
int8_bytes = sum(p.stat().st_size for p in MODELS.glob("model_int8.onnx*"))
results["sizes"] = {"fp32_onnx_bytes": fp32_bytes, "int8_onnx_bytes": int8_bytes,
                    "fp32_onnx_gb": round(fp32_bytes / 1024**3, 3),
                    "int8_onnx_mb": round(int8_bytes / 1024**2, 1),
                    "compression_ratio": round(fp32_bytes / int8_bytes, 2)}
results["quantization"] = {"method": "quantize_dynamic", "weight_type": "QUInt8",
                           "extra_options": {"WeightSymmetric": True},
                           "use_external_data_format": True, "wall_s": round(q_s, 1)}
print(json.dumps(results["sizes"]), flush=True)

print("benchmarking fp32 ONNX (CPU)...", flush=True)
r32, e32 = bench("fp32_onnx_cpu", FP32)
print("benchmarking int8 ONNX (CPU)...", flush=True)
r8, e8 = bench("int8_onnx_cpu", INT8)
results["benchmark"] = {"fp32_onnx_cpu": r32, "int8_onnx_cpu": r8}
results["speedup"] = {"texts_s_bs8": round(r8["bs8"]["texts_s"] / r32["bs8"]["texts_s"], 2),
                      "texts_s_bs32": round(r8["bs32"]["texts_s"] / r32["bs32"]["texts_s"], 2),
                      "latency_p50": round(r32["latency_ms"]["p50"] / r8["latency_ms"]["p50"], 2)}

print("measuring quality vs PyTorch fp32 ground truth...", flush=True)
results["quality"] = {"fp32_onnx_vs_torch_fp32": quality(e32),
                      "int8_vs_torch_fp32": quality(e8)}

out = DATA / "quantize_results.json"
out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print(f"saved -> {out}")
