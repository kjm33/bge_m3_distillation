"""Verify determinism of OpenRouter baai/bge-m3 embeddings and compare with local fp32.

Tests:
  T1  repeat      : same batch x5 runs -> bitwise / max-abs-diff / cosine
  T2  delayed     : re-run after delay -> same comparisons
  T3  batch-vs-1  : each text individually vs batch result
  T4  providers   : pinned DeepInfra (fp32) vs pinned Parasail (unknown quant) vs default
  T5  local fp32  : SentenceTransformers BAAI/bge-m3 (CLS + mean pooling) vs API
  T6  in-batch dup: same text twice in one request -> identical vectors
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
load_dotenv(HERE.parents[1] / ".env")

API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "baai/bge-m3"
KEY = os.environ["OPENROUTER_API_KEY"]
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Sztuczna inteligencja zmienia sposób, w jaki pracujemy.",
    "机器学习是人工智能的一个分支。",
    "Künstliche Intelligenz verändert die Welt.",
    "Машинное обучение —分支 of ИИ.",  # mixed scripts
    "a",  # single char
    "🚀🌟 embedding edge case with emoji 🌟🚀",
    "𝕦𝕟𝕚𝕔𝕠𝕕𝕖 𝕥𝕖𝕩𝕥 𝕨𝕚𝕥𝕙 mathematical alphanumeric symbols",
    ("Embedding models map text to dense vectors. BGE-M3 is a multilingual model "
     "supporting up to 8192 tokens, dense, sparse and multi-vector representations. "
     "It is trained with self-knowledge distillation on more than 100 languages. " * 8),
    "dup-text-marker-should-be-identical-inside-a-batch",  # also sent twice (T6)
]

# real instructions from the instruction_extraction project (one per top language)
import collections
import json as _json
_in_path = DATA / "instructions_v3.jsonl"
if _in_path.exists():
    _by_lang = collections.defaultdict(list)
    with open(_in_path) as _f:
        for _line in _f:
            _d = _json.loads(_line)
            _by_lang[_d.get("language", "?")].append(_d["text"])
    for _lang in ["en", "de", "fr", "pl", "es"]:
        _cand = [t for t in _by_lang.get(_lang, []) if 60 < len(t) < 250]
        if _cand:
            TEXTS.append(_cand[len(_cand) // 2])  # deterministic pick

session = requests.Session()
total_tokens = 0


def embed(texts, provider=None, retries=4):
    global total_tokens
    body = {"model": MODEL, "input": texts}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    last = None
    for attempt in range(retries):
        r = session.post(API_URL, headers=HDRS, json=body, timeout=120)
        if r.status_code == 200:
            j = r.json()
            total_tokens += j.get("usage", {}).get("total_tokens", 0)
            vecs = [np.asarray(d["embedding"], dtype=np.float64) for d in j["data"]]
            return vecs, j
        last = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in (429, 529, 503):
            time.sleep(2**attempt * 2)
            continue
        break
    raise RuntimeError(f"request failed provider={provider}: {last}")


def cmp(a, b):
    an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return {
        "bitwise_equal": bool(np.array_equal(a, b)),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "cosine": float(np.dot(an, bn)),
    }


def agg(pairs):
    return {
        "all_bitwise": all(p["bitwise_equal"] for p in pairs),
        "max_abs_diff": max(p["max_abs_diff"] for p in pairs),
        "min_cosine": min(p["cosine"] for p in pairs),
    }


res = {"model": MODEL, "n_texts": len(TEXTS)}

# T1 repeat x5
runs, metas = [], None
for i in range(5):
    v, metas = embed(TEXTS)
    runs.append(v)
    print(f"T1 run {i+1}/5 ok")
t1 = [agg([cmp(runs[0][j], runs[i][j]) for j in range(len(TEXTS))]) for i in range(1, 5)]
res["T1_repeat_x5"] = t1
res["T1_response_meta_keys"] = sorted(metas.keys()) if metas else []
if isinstance(metas, dict):
    res["T1_provider_field"] = metas.get("provider")

# T2 delayed retry
time.sleep(20)
v2, _ = embed(TEXTS)
res["T2_delayed_20s"] = agg([cmp(runs[0][j], v2[j]) for j in range(len(TEXTS))])
print("T2 delayed ok")

# T3 batch vs single
singles = []
for t in TEXTS:
    sv, _ = embed(t)
    singles.append(sv[0])
res["T3_batch_vs_single"] = agg([cmp(runs[0][j], singles[j]) for j in range(len(TEXTS))])
print("T3 singles ok")

# T4 provider pinned
prov_res = {}
for prov in ["deepinfra", "parasail"]:
    try:
        pv, pj = embed(TEXTS, provider=prov)
        prov_res[prov] = {
            "vs_default": agg([cmp(runs[0][j], pv[j]) for j in range(len(TEXTS))]),
            "norm_range": [float(min(np.linalg.norm(x) for x in pv)),
                           float(max(np.linalg.norm(x) for x in pv))],
        }
    except Exception as e:
        prov_res[prov] = {"error": str(e)[:300]}
if "deepinfra" in prov_res and "parasail" in prov_res and "error" not in prov_res["deepinfra"] and "error" not in prov_res["parasail"]:
    pass  # filled after T4 rerun below
res["T4_providers"] = prov_res
print("T4 providers ok")

# T5 local fp32 comparison (CLS + mean pooling)
from transformers import AutoTokenizer, AutoModel
import torch

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
mdl = AutoModel.from_pretrained("BAAI/bge-m3").cuda().eval()
api = np.stack([runs[0][j] for j in range(len(TEXTS))])
api_n = api / np.linalg.norm(api, axis=1, keepdims=True)
with torch.no_grad():
    enc = tok(TEXTS, padding=True, truncation=True, max_length=8192, return_tensors="pt").to("cuda")
    h = mdl(**enc).last_hidden_state[0] if False else mdl(**enc).last_hidden_state
cls = h[:, 0, :].cpu().numpy()
mask = enc["attention_mask"].cpu().numpy()[:, :, None]
mean = ((h.cpu().numpy() * mask).sum(1) / mask.sum(1)).astype(np.float32)
cls_n = cls / np.linalg.norm(cls, axis=1, keepdims=True)
mean_n = mean / np.linalg.norm(mean, axis=1, keepdims=True)
res["T5_local_fp32"] = {
    "api_norm_range": [float(np.linalg.norm(api, axis=1).min()), float(np.linalg.norm(api, axis=1).max())],
    "cos_api_vs_local_cls": [float(np.dot(api_n[i], cls_n[i])) for i in range(len(TEXTS))],
    "cos_api_vs_local_mean": [float(np.dot(api_n[i], mean_n[i])) for i in range(len(TEXTS))],
}
print("T5 local ok")

# T6 in-batch duplicates
dup_texts = [TEXTS[-1], TEXTS[-1], TEXTS[0], TEXTS[0]]
dv, _ = embed(dup_texts)
res["T6_in_batch_dups"] = {
    "dup_pair_bitwise": bool(np.array_equal(dv[0], dv[1])),
    "other_pair_bitwise": bool(np.array_equal(dv[2], dv[3])),
    "dup_cmp": cmp(dv[0], dv[1]),
}

res["usage_total_tokens_all_requests"] = total_tokens
res["est_cost_usd"] = total_tokens * 0.00000001
out = DATA / "determinism_results.json"
out.write_text(json.dumps(res, indent=2))
print(f"\nsaved -> {out}\n")


def fmt(a):
    return f"bitwise={a['all_bitwise']}  maxabs={a['max_abs_diff']:.3e}  mincos={a['min_cosine']:.6f}"


print("=== SUMMARY ===")
print(f"T1 repeat x5 (4 comparisons vs run0):")
for i, a in enumerate(t1):
    print(f"   run{i+2}: {fmt(a)}")
print(f"T2 delayed 20s : {fmt(res['T2_delayed_20s'])}")
print(f"T3 batch vs 1  : {fmt(res['T3_batch_vs_single'])}")
for p, d in prov_res.items():
    print(f"T4 {p:10s}: {'ERROR: ' + d['error'][:80] if 'error' in d else fmt(d['vs_default'])}")
c5c = res["T5_local_fp32"]["cos_api_vs_local_cls"]
c5m = res["T5_local_fp32"]["cos_api_vs_local_mean"]
print(f"T5 API vs local CLS  : min={min(c5c):.6f} mean={np.mean(c5c):.6f}")
print(f"T5 API vs local MEAN : min={min(c5m):.6f} mean={np.mean(c5m):.6f}")
print(f"T5 API norms range   : {res['T5_local_fp32']['api_norm_range']}")
print(f"T6 dup bitwise       : {res['T6_in_batch_dups']['dup_pair_bitwise']}")
print(f"total tokens={total_tokens}  est cost=${res['est_cost_usd']:.6f}")
