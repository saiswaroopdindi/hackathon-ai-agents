"""Deterministic answer-quality checks shared by agents (self-check), the Critic and the evaluator."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

from core.text import tokenize

CITE_RE = re.compile(r"\[([A-Z]\d{1,3})\]")


def _units(answer: str) -> List[str]:
    return [u.strip() for u in re.split(r"\n+|(?<=[\.\!\?])\s+(?=[A-Z])", answer)
            if len(u.strip()) > 40 and not u.strip().startswith(("#", "|---", "| Priority"))]


def response_metrics(answer: str, evidence: Iterable[Dict], query: str) -> Dict:
    """evidence: dicts with 'cid' and 'text'."""
    ev = {e["cid"]: e for e in evidence if e.get("cid")}
    units = _units(answer)
    cited = [u for u in units if CITE_RE.search(u)]
    all_cites = CITE_RE.findall(answer)
    valid = [c for c in all_cites if c in ev]
    grounded = 0
    for u in cited:
        ut = set(tokenize(CITE_RE.sub("", u)))
        best = max((len(ut & set(tokenize(str(ev[c].get("text", ""))))) / max(len(ut), 1)
                    for c in CITE_RE.findall(u) if c in ev), default=0)
        grounded += best >= 0.3
    qt = set(tokenize(query))
    return {"citation_coverage": round(len(cited) / max(len(units), 1), 3),
            "citation_validity": round(len(valid) / max(len(all_cites), 1), 3) if all_cites else 0.0,
            "grounding_rate": round(grounded / max(len(cited), 1), 3) if cited else 0.0,
            "query_coverage": round(len(qt & set(tokenize(answer))) / max(len(qt), 1), 3),
            "invalid_citations": sorted(set(c for c in all_cites if c not in ev)),
            "n_citations": len(all_cites)}


def self_check(answer: str, evidence: Iterable[Dict], query: str, next_actions=None,
               hypotheses: bool = False) -> Dict:
    ev = list(evidence)
    m = response_metrics(answer, ev, query)
    issues = []
    if not ev:
        issues.append("No evidence was gathered - call a search tool before answering.")
    if m["invalid_citations"]:
        issues.append(f"Citations not in the evidence: {', '.join(m['invalid_citations'])} - remove or replace them.")
    if ev and m["n_citations"] == 0:
        issues.append("The answer has no citations - cite the passages that support each claim.")
    elif ev and m["citation_coverage"] < 0.4:
        issues.append(f"Only {m['citation_coverage']:.0%} of substantive sentences are cited.")
    if m["n_citations"] and m["grounding_rate"] < 0.5:
        issues.append("Several cited sentences do not match the wording of their passages - check the claims.")
    if next_actions is not None and (not next_actions or not all(a.get("kpi") for a in next_actions)):
        issues.append("Provide next actions, each with a measurable KPI.")
    if hypotheses and "hypothes" not in answer.lower():
        issues.append("Fault hypotheses exist but the answer does not label them as hypotheses vs confirmed facts.")
    return {"passed": not issues, "issues": issues, "metrics": m}
