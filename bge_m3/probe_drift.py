"""Probe source of OpenRouter bge-m3 nondeterminism: which texts drift, dup position effect."""
import os
from pathlib import Path
import numpy as np
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")
KEY = os.environ["OPENROUTER_API_KEY"]
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
URL = "https://openrouter.ai/api/v1/embeddings"

import json as _json
texts = []
with open(HERE / "data" / "instructions_v3.jsonl") as f:
    for line in f:
        d = _json.loads(line)
        if d.get("language") == "en" and 60 < len(d["text"]) < 250:
            texts.append(d["text"])
        if len(texts) >= 8:
            break
texts.append("a")
texts.append("机器学习是人工智能的一个分支。")

def embed(inp, provider=None):
    body = {"model": "baai/bge-m3", "input": inp}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    r = requests.post(URL, headers=HDRS, json=body, timeout=120)
    r.raise_for_status()
    return [np.asarray(d["embedding"], dtype=np.float64) for d in r.json()["data"]]

def diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))), float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

n = len(texts)
A, B, C = embed(texts), embed(texts), embed(texts)
print("=== 3 identical batch runs, per-text max|diff| run1 vs run2 / run3 ===")
for i, t in enumerate(texts):
    d12, c12 = diff(A[i], B[i]); d13, c13 = diff(A[i], C[i])
    mark = " <-- DRIFTS" if max(d12, d13) > 1e-6 else ""
    print(f"[{i:2d}] len={len(t):3d} d12={d12:.2e} d13={d13:.2e} cos={min(c12,c13):.8f}{mark}  {t[:40]!r}")

print("\n=== dup within batch: same text at positions 0,1,7 ===")
dup = embed([texts[3]] * 2 + [texts[4]] + [texts[5]] + [texts[6]] + [texts[7]] + [texts[3]])
for j in [1, 6]:
    d, c = diff(dup[0], dup[j])
    print(f"pos0 vs pos{j}: maxabs={d:.2e} cos={c:.10f}")

print("\n=== cross-provider per-text (deepinfra vs parasail) ===")
D, P = embed(texts, provider="deepinfra"), embed(texts, provider="parasail")
for i in range(n):
    d, c = diff(D[i], P[i])
    print(f"[{i:2d}] DI vs PS: maxabs={d:.2e} cos={c:.9f}")
dA, cA = diff(D[0], A[0]); dP, cP = diff(P[0], A[0])
print(f"default-vs-DI {dA:.2e}/{cA:.10f}   default-vs-PS {dP:.2e}/{cP:.10f}")
