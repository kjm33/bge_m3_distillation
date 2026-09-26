"""Local SQLite cache for OpenRouter bge-m3 embeddings + cold-vs-warm benchmark.

Key = sha256(model | provider | NFC+stripped text). Vectors are L2-normalized
1024-dim -> stored as float32 blob (4 KB/vector). An int8 storage variant
(1 KB + 4 B scale) is benchmarked for space-constrained use.
"""
import hashlib
import json
import os
import sqlite3
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
load_dotenv(HERE.parents[1] / ".env")

MODEL = "baai/bge-m3"
PROVIDER = None  # None = default routing; pin e.g. "deepinfra" for stricter consistency
URL = "https://openrouter.ai/api/v1/embeddings"


def norm_text(t: str) -> str:
    return unicodedata.normalize("NFC", t).strip()


def cache_key(model: str, provider: str | None, text: str) -> str:
    return hashlib.sha256(f"{model}|{provider or 'default'}|{norm_text(text)}".encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, db_path: Path, model: str = MODEL, provider: str | None = PROVIDER):
        self.db_path, self.model, self.provider = db_path, model, provider
        self.con = sqlite3.connect(db_path)
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS embeddings(
                 key TEXT PRIMARY KEY, model TEXT, provider TEXT, dim INTEGER,
                 vec BLOB, created_at REAL, hits INTEGER DEFAULT 0)"""
        )
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)")
        self.con.commit()
        self.api_calls = 0
        self.cache_hits = 0
        self.tokens_billed = 0

    def _get_many(self, texts):
        out, missing = {}, []
        for t in texts:
            k = cache_key(self.model, self.provider, t)
            row = self.con.execute(
                "SELECT vec, dim FROM embeddings WHERE key=?", (k,)).fetchone()
            if row:
                v = np.frombuffer(row[0], dtype=np.float32).astype(np.float64)
                out[t] = v
                self.con.execute("UPDATE embeddings SET hits=hits+1 WHERE key=?", (k,))
            else:
                missing.append((k, t))
        self.con.commit()
        return out, missing

    def _put(self, items):
        self.con.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?,0)",
            [(k, self.model, self.provider or "default", len(v),
              np.asarray(v, dtype=np.float32).tobytes(), time.time())
             for k, v in items])

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return embeddings for texts, using cache; call API only for misses."""
        uniq = list(dict.fromkeys(norm_text(t) for t in texts))
        got, missing = self._get_many([norm_text(t) for t in texts])
        self.cache_hits += len(uniq) - len(missing)
        if missing:
            key = os.environ["OPENROUTER_API_KEY"]
            for i in range(0, len(missing), batch_size):
                chunk = missing[i:i + batch_size]
                body = {"model": self.model, "input": [t for _, t in chunk]}
                if self.provider:
                    body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
                r = requests.post(URL, headers={
                    "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=body, timeout=120)
                r.raise_for_status()
                j = r.json()
                self.api_calls += 1
                self.tokens_billed += j.get("usage", {}).get("total_tokens", 0)
                pairs = [(chunk[idx][0], np.asarray(d["embedding"], dtype=np.float64))
                         for idx, d in enumerate(j["data"])]
                self._put(pairs)
                for k, v in pairs:
                    got[[t for kk, t in chunk if kk == k][0]] = v
        self.con.commit()
        return np.stack([got[norm_text(t)] for t in texts])

    def stats(self):
        n, dim = self.con.execute(
            "SELECT COUNT(*), MAX(dim) FROM embeddings").fetchone()
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"vectors": n, "dim": dim, "db_bytes": size,
                "bytes_per_vector": (size / n) if n else 0,
                "api_calls": self.api_calls, "cache_hits": self.cache_hits,
                "tokens_billed": self.tokens_billed}


def load_instructions(n=200, path="/home/kamil/projects/here/instruction_extraction/data/instructions_v3.jsonl"):
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(d["text"])
            if len(out) >= n:
                break
    return out


if __name__ == "__main__":
    texts = load_instructions(200)
    db = DATA / "embeddings_cache.db"
    if db.exists():
        db.unlink()
    cache = EmbeddingCache(db)

    t0 = time.perf_counter()
    v1 = cache.embed(texts)
    cold = time.perf_counter() - t0
    cold_stats = cache.stats()

    t0 = time.perf_counter()
    v2 = cache.embed(texts)
    warm = time.perf_counter() - t0
    warm_stats = cache.stats()

    # pure sqlite lookup latency (single text, median of 100)
    lat = []
    for t in texts[:100]:
        t0 = time.perf_counter()
        cache.embed([t])
        lat.append(time.perf_counter() - t0)
    lookup_ms = float(np.median(lat)) * 1000

    # identical texts re-fetched from cache: compare vs first fetch
    assert np.array_equal(v1, v2), "cache must return identical vectors"

    # int8 storage variant: per-vector scale, cosine vs fp32
    v32 = v1.astype(np.float32)
    scale = np.abs(v32).max(axis=1) / 127.0
    v8 = np.round(v32 / scale[:, None]).astype(np.int8)
    recon = (v8.astype(np.float32) * scale[:, None])
    cos = np.sum(recon * v32, axis=1) / (
        np.linalg.norm(recon, axis=1) * np.linalg.norm(v32, axis=1))

    # retrieval sanity: nearest-neighbour agreement fp32 vs int8-stored
    q = v32[:20]; D32 = v32 @ v32.T; D8 = recon @ recon.T
    nn32 = np.argsort(-D32, axis=1)[:, 1]
    nn8 = np.argsort(-D8, axis=1)[:, 1]
    agree = float((nn32 == nn8).mean())

    report = {
        "n_texts": len(texts),
        "cold_s": round(cold, 2), "cold_ms_per_text": round(cold / len(texts) * 1000, 1),
        "warm_s": round(warm, 4), "warm_ms_per_text": round(warm / len(texts) * 1000, 3),
        "sqlite_single_lookup_ms": round(lookup_ms, 3),
        "speedup": round(cold / warm, 1),
        "cold": cold_stats, "warm": warm_stats,
        "int8_storage": {
            "bytes_per_vector_fp32": 4096, "bytes_per_vector_int8": 1028,
            "min_cosine_vs_fp32": float(cos.min()), "nn_agreement_top1": agree},
    }
    print(json.dumps(report, indent=2))
    (DATA / "cache_benchmark.json").write_text(json.dumps(report, indent=2))
