"""Profile local BGE-M3 inference: disk, load time, RAM, VRAM, latency, throughput.

Configs: CPU fp32, GPU fp32, GPU fp16 (RTX 3090). Workload: 512 real instructions
from the instruction_extraction project (saved to data/bench_texts.json, reused by
quantize_compare.py). Ground-truth fp32 GPU embeddings saved for Phase 3.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL_ID = "BAAI/bge-m3"

# ---------- bench set ----------
src = DATA / "instructions_v3.jsonl"
by_lang: dict[str, list[str]] = {}
with open(src) as f:
    for line in f:
        d = json.loads(line)
        by_lang.setdefault(d.get("language", "?"), []).append(d["text"])
texts = []
i = 0
while len(texts) < 512:
    for lang in ["en", "de", "fr", "pl", "es"]:
        pool = by_lang[lang]
        texts.append(pool[i % len(pool)])
    i += 1
texts = texts[:512]
(DATA / "bench_texts.json").write_text(json.dumps(texts))

proc = psutil.Process(os.getpid())
tok = AutoTokenizer.from_pretrained(MODEL_ID)


def n_tokens(ts):
    enc = tok(ts, padding=False, truncation=False)
    return sum(len(x) for x in enc["input_ids"])


def rss_mb():
    return proc.memory_info().rss / 1024**2


def load_model(device, dtype=torch.float32):
    t0 = time.perf_counter()
    m = AutoModel.from_pretrained(MODEL_ID).to(device=device, dtype=dtype).eval()
    load_s = time.perf_counter() - t0
    return m, load_s


def encode(model, device, ts, bs=32, amp=False):
    out = []
    with torch.no_grad():
        for i in range(0, len(ts), bs):
            chunk = ts[i:i + bs]
            enc = tok(chunk, padding=True, truncation=True, max_length=8192,
                      return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", enabled=amp):
                h = model(**enc).last_hidden_state[:, 0, :]
            out.append(h.float().cpu().numpy())
    return np.concatenate(out)


def bench(tag, device, dtype=torch.float32, amp=False):
    m, load_s = load_model(device, dtype)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    base_rss = rss_mb()
    encode(m, device, texts[:32], amp=amp)  # warmup
    r = {"load_s": round(load_s, 2), "config": tag, "device": device,
         "dtype": str(dtype).replace("torch.", "")}

    # throughput at several batch sizes
    toks = n_tokens(texts)
    for bs in [1, 8, 32, 128]:
        if device == "cpu" and bs > 32:
            continue
        subset = texts[:256] if device == "cpu" else texts
        stoks = n_tokens(subset)
        t0 = time.perf_counter()
        encode(m, device, subset, bs=bs, amp=amp)
        dt = time.perf_counter() - t0
        r[f"bs{bs}"] = {"texts_s": round(len(subset) / dt, 1),
                        "tokens_s": round(stoks / dt, 0),
                        "ms_per_batch": round(dt / (len(subset) / bs) * 1000, 1)}

    # single-text latency p50/p95 (20 sequential requests)
    lat = []
    for t in texts[:20]:
        t0 = time.perf_counter()
        encode(m, device, [t], bs=1, amp=amp)
        lat.append((time.perf_counter() - t0) * 1000)
    r["latency_ms"] = {"p50": round(float(np.median(lat)), 1),
                       "p95": round(float(np.percentile(lat, 95)), 1)}

    # peak resources incl. inference
    encode(m, device, texts, bs=32, amp=amp)
    r["peak_ram_mb"] = round(rss_mb(), 0)  # process RSS total (incl. python+torch)
    r["rss_delta_mb"] = round(rss_mb() - base_rss + 50, 0)  # approx model+act footprint
    if device == "cuda":
        r["peak_vram_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 0)
    # long doc (8192 tokens) single pass
    long_doc = "emb " * 5400  # ~8100 tokens
    lt = n_tokens([long_doc])
    t0 = time.perf_counter()
    encode(m, device, [long_doc], bs=1, amp=amp)
    r["long_doc"] = {"tokens": lt,
                     "s": round(time.perf_counter() - t0, 2),
                     "tokens_s": round(lt / (time.perf_counter() - t0 + 1e-9), 0)}
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return r


results = {"model": MODEL_ID, "n_bench_texts": len(texts),
           "bench_tokens": n_tokens(texts), "avg_tokens_per_text": round(n_tokens(texts) / len(texts), 1)}

# disk usage (fp32 weights on disk)
hub = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3"
total = sum(f.stat().st_size for f in hub.rglob("*") if f.is_file())
onnx = sum(f.stat().st_size for f in (hub / "snapshots").rglob("onnx/*") if f.is_file())
results["disk"] = {"weights_fp32_gb": round(2271145830 / 1024**3, 2),
                   "onnx_fp32_gb": round(onnx / 1024**3, 2),
                   "total_cache_gb": round(total / 1024**3, 2)}

results["cpu_fp32_threads"] = torch.get_num_threads()
results["cpu_fp32"] = bench("CPU fp32", "cpu")
results["gpu_fp32"] = bench("GPU fp32", "cuda", torch.float32)
results["gpu_fp16"] = bench("GPU fp16 (model.half)", "cuda", torch.float16)

# ground truth for phase 3: GPU fp32 embeddings of bench set
m, _ = load_model("cuda", torch.float32)
gt = encode(m, "cuda", texts, bs=64)
np.save(DATA / "groundtruth_fp32.npy", gt.astype(np.float32))
del m
torch.cuda.empty_cache()

print(json.dumps(results, indent=2))
(DATA / "profile_results.json").write_text(json.dumps(results, indent=2))
