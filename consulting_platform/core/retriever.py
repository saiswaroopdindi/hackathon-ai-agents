"""Hybrid lexical retriever: BM25 (unigrams) + phrase BM25 (bigrams) + metadata filtering.

Implemented from scratch with numpy so the project has no heavy ML dependencies.
If `sentence-transformers` is installed, a dense re-ranker is applied automatically.
"""
from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config
from core.text import bigrams, tokenize


@dataclass
class Hit:
    doc: Dict
    score: float
    rank: int = 0
    cid: str = ""            # citation id, e.g. S1

    @property
    def citation(self) -> str:
        d = self.doc
        page = f", p.{d['page']}" if d.get("doc_type") not in ("clinical trial", "catalog", "web") else ""
        return f"{d.get('source','')}{page}"

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.doc.items() if k != "tokens"}
        return {"cid": self.cid, "score": round(float(self.score), 4), "citation": self.citation, **d}


class BM25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.72):
        self.k1, self.b = k1, b
        self.docs: List[Dict] = []
        self.post: Dict[str, tuple] = {}
        self.post_bi: Dict[str, tuple] = {}
        self.idf: Dict[str, float] = {}
        self.idf_bi: Dict[str, float] = {}
        self.dl = None
        self.avgdl = 1.0

    # ---------------------------------------------------------------- build
    def fit(self, docs: List[Dict]) -> "BM25Index":
        self.docs = docs
        n = len(docs)
        uni, bi = defaultdict(list), defaultdict(list)
        lengths = np.zeros(n, dtype=np.float32)
        for i, d in enumerate(docs):
            toks = tokenize(d["source"] + " " + d["text"]) if d.get("doc_type") in ("catalog", "clinical trial") \
                else tokenize(d["text"])
            lengths[i] = len(toks)
            for t, c in Counter(toks).items():
                uni[t].append((i, c))
            for t, c in Counter(bigrams(toks)).items():
                bi[t].append((i, c))
        self.dl = lengths
        self.avgdl = float(lengths.mean()) if n else 1.0

        def pack(src):
            out, idf = {}, {}
            for t, lst in src.items():
                ids = np.fromiter((x[0] for x in lst), dtype=np.int32, count=len(lst))
                tfs = np.fromiter((x[1] for x in lst), dtype=np.float32, count=len(lst))
                out[t] = (ids, tfs)
                df = len(lst)
                idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
            return out, idf

        self.post, self.idf = pack(uni)
        self.post_bi, self.idf_bi = pack(bi)
        return self

    # ---------------------------------------------------------------- query
    def _accumulate(self, scores, terms, post, idf, weight):
        norm = self.k1 * (1 - self.b + self.b * self.dl / self.avgdl)
        for t, qtf in Counter(terms).items():
            if t not in post:
                continue
            ids, tfs = post[t]
            scores[ids] += weight * idf[t] * (tfs * (self.k1 + 1)) / (tfs + norm[ids])

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.docs), dtype=np.float32)
        if not q:
            return s
        self._accumulate(s, q, self.post, self.idf, 1.0)
        self._accumulate(s, bigrams(q), self.post_bi, self.idf_bi, 0.6)
        return s


class Retriever:
    """One BM25 index per knowledge base, loaded lazily from data/index."""

    _cache: Dict[str, BM25Index] = {}
    _dense = None

    def __init__(self, name: str):
        self.name = name
        if name not in Retriever._cache:
            p = config.INDEX_DIR / f"{name}.pkl"
            if not p.exists():
                # first run on a fresh clone: build this knowledge base automatically
                print(f"[setup] building '{name}' index (first run only) ...", flush=True)
                build_all([name], verbose=True)
            if not p.exists():
                raise FileNotFoundError(f"Index '{name}' could not be built - check DATA_ROOT ({config.DATA_ROOT})")
            with open(p, "rb") as fh:
                Retriever._cache[name] = pickle.load(fh)
        self.index = Retriever._cache[name]

    @property
    def size(self) -> int:
        return len(self.index.docs)

    def search(self, query: str, k: int = config.TOP_K, filters: Optional[Dict[str, List[str]]] = None,
               boost: Optional[Dict[str, List[str]]] = None, per_source: int = 3,
               min_score: float = 0.0) -> List[Hit]:
        s = self.index.scores(query)
        docs = self.index.docs
        if filters:
            # hard filter only if it leaves something; otherwise fall back to soft boost
            mask = np.ones(len(docs), dtype=bool)
            for key, vals in filters.items():
                vals_l = [v.lower() for v in vals]
                mask &= np.array([str(d.get(key, "")).lower() in vals_l for d in docs])
            if mask.any() and (s[mask] > 0).any():
                s = np.where(mask, s, 0)
        if boost:
            for key, vals in boost.items():
                vals_l = [v.lower() for v in vals]
                b = np.array([1.25 if str(d.get(key, "")).lower() in vals_l else 1.0 for d in docs])
                s = s * b
        order = np.argsort(-s)[: max(k * 8, 40)]
        hits, seen = [], Counter()
        for i in order:
            if s[i] <= min_score:
                break
            src = docs[i].get("path", "") + str(docs[i].get("nct", ""))
            if seen[src] >= per_source:            # diversity across documents
                continue
            seen[src] += 1
            hits.append(Hit(doc=docs[i], score=float(s[i])))
        hits = self._dense_rerank(query, hits)[:k]
        for r, h in enumerate(hits, 1):
            h.rank = r
        return hits

    # optional dense re-ranking ------------------------------------------------
    @classmethod
    def _dense_model(cls):
        if cls._dense is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                cls._dense = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                cls._dense = False
        return cls._dense

    def _dense_rerank(self, query: str, hits: List[Hit]) -> List[Hit]:
        model = self._dense_model()
        if not model or len(hits) < 2:
            return hits
        texts = [h.doc["text"][:1000] for h in hits]
        emb = model.encode([query] + texts, normalize_embeddings=True)
        cos = emb[1:] @ emb[0]
        bm = np.array([h.score for h in hits])
        bm = bm / (bm.max() or 1)
        fused = 0.5 * bm + 0.5 * cos
        for h, f in zip(hits, fused):
            h.score = float(f)
        return sorted(hits, key=lambda h: -h.score)


def build_all(names=None, verbose=True) -> Dict[str, int]:
    from core.ingest import BUILDERS
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, fn in BUILDERS.items():
        if names and name not in names:
            continue
        t = time.time()
        docs = fn()
        idx = BM25Index().fit(docs)
        with open(config.INDEX_DIR / f"{name}.pkl", "wb") as fh:
            pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        stats[name] = len(docs)
        if verbose:
            n_src = len({d.get('path') for d in docs})
            print(f"  [{name:10s}] {len(docs):6d} chunks from {n_src} sources  ({time.time()-t:.1f}s)")
    sp = config.INDEX_DIR / "stats.json"
    old = json.loads(sp.read_text()) if sp.exists() else {}
    old.update(stats)
    sp.write_text(json.dumps(old, indent=2))
    for n in stats:
        Retriever._cache.pop(n, None)
    return stats
