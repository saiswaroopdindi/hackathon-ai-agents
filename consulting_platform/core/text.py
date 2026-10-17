"""Text utilities: tokenisation, light stemming, chunking, sentence splitting."""
from __future__ import annotations

import re
from typing import Iterable, List

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also may might must shall within without per via using use used e g
etc us please tell give show list explain describe what's which whats should would like want need
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.\/]*[a-z0-9]|[a-z0-9]")


def stem(tok: str) -> str:
    """Very small suffix stripper (Porter-lite). Keeps model numbers intact."""
    if any(c.isdigit() for c in tok) or len(tok) <= 4:
        return tok
    for suf in ("ational", "ization", "fulness", "iveness", "ations", "ments", "ities",
                "ing", "edly", "ied", "ies", "ers", "ed", "es", "ly", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            if suf in ("ied", "ies"):
                return tok[: -len(suf)] + "y"
            return tok[: -len(suf)]
    return tok


def tokenize(text: str, keep_stop: bool = False) -> List[str]:
    text = text.lower()
    toks = _TOKEN_RE.findall(text)
    out = []
    for t in toks:
        t = t.strip(".-/")
        if not t or (not keep_stop and t in STOPWORDS):
            continue
        out.append(stem(t))
        # split hyphenated/compound tokens too (e.g. "isr4331/k9" -> isr4331, k9)
        if any(c in t for c in "-/."):
            for p in re.split(r"[-/.]", t):
                if p and p != t and p not in STOPWORDS and len(p) > 1:
                    out.append(stem(p))
    return out


def bigrams(tokens: List[str]) -> List[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)            # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split on paragraph/sentence boundaries into ~size-char windows with overlap."""
    text = clean_text(text)
    if len(text) <= size:
        return [text] if len(text) > 80 else []
    pieces = re.split(r"(?<=[\.\?\!])\s+|\n\n", text)
    chunks, cur = [], ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur} {p}".strip()
        else:
            if len(cur) > 80:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            # start new chunk with overlap tail (cut at word boundary)
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = f"{tail} {p}".strip()
            while len(cur) > size * 1.5:           # very long run-on (tables)
                chunks.append(cur[:size])
                cur = cur[size - overlap:]
    if len(cur) > 80:
        chunks.append(cur)
    return chunks


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\(\"])", text)
    return [s.strip() for s in sents if 25 <= len(s.strip()) <= 600]


def overlap_score(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)
