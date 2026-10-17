"""Tools that specialist agents can call inside their reasoning loop.

Each tool wraps a retrieval or recommender capability, records every passage it returns in the agent's
evidence registry (so citations like [T3] stay stable across steps), and returns a compact text observation.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.retriever import Hit, Retriever
from core.text import tokenize


# ============================================================================ context
@dataclass
class AgentContext:
    domain: str
    prefix: str
    hits: List[Hit] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)
    extras: Dict = field(default_factory=dict)
    missing_info: List[str] = field(default_factory=list)
    _keys: Dict[str, str] = field(default_factory=dict)

    def add(self, h: Hit) -> str:
        d = h.doc
        key = f"{d.get('path')}|{d.get('page')}|{d.get('chunk')}|{d.get('nct', '')}"
        if key in self._keys:
            h.cid = self._keys[key]
            return h.cid
        h.cid = f"{self.prefix}{len(self.hits) + 1}"
        self._keys[key] = h.cid
        self.hits.append(h)
        return h.cid

    @property
    def evidence_ids(self):
        return {h.cid for h in self.hits}

    def add_recs(self, recs: List[Dict]):
        seen = {r["title"] for r in self.recommendations}
        for r in recs:
            if r["title"] not in seen:
                self.recommendations.append(r)
                seen.add(r["title"])


@dataclass
class Tool:
    name: str
    description: str
    params: Dict[str, str]
    fn: Callable

    def spec(self) -> str:
        p = ", ".join(f"{k}: {v}" for k, v in self.params.items()) or "no arguments"
        return f"- {self.name}({p})\n    {self.description}"

    def __call__(self, ctx: AgentContext, args: Dict) -> str:
        args = {k: v for k, v in (args or {}).items() if k in self.params}
        return self.fn(ctx, **args)


def _fmt_hits(ctx: AgentContext, hits: List[Hit], chars: int = 650) -> str:
    if not hits:
        return "No matching passages found. Try different keywords or remove filters."
    lines = []
    for h in hits:
        cid = ctx.add(h)
        txt = " ".join(h.doc["text"].split())[:chars]
        lines.append(f"[{cid}] {h.citation} (score {h.score:.1f}): {txt}")
    return "\n".join(lines)


def _as_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [s.strip() for s in re.split(r"[,;/]| and ", str(v)) if s.strip()]


# ============================================================================ technology
def technology_tools() -> List[Tool]:
    from core.agents import COMPONENT_CATEGORIES, FAULT_RULES, TechnologyAgent
    agent = TechnologyAgent(None)

    def search_tech_docs(ctx, query: str, company: str = ""):
        boost = {"company": [company]} if company else None
        hits = Retriever("technology").search(agent.expand(query), k=5, boost=boost)
        return _fmt_hits(ctx, hits)

    def identify_router_model(ctx, text: str):
        m = agent.detect_model(text)
        if not m:
            ctx.missing_info.append("Exact router model / PID")
            return "No Cisco model number found. Ask the client for the exact model/PID (e.g. ISR4331/K9)."
        return json.dumps({"model": m[0], "family": m[1]})

    def diagnose_symptoms(ctx, symptoms: str):
        ql = symptoms.lower()
        hyps = []
        for r in FAULT_RULES:
            matched = [t for t in r["triggers"] if t in ql]
            if matched:
                hyps.append({"hypothesis": r["name"], "matched_symptoms": matched,
                             "likelihood": round(min(0.2 + 0.18 * len(matched), 0.85), 2),
                             "status": "HYPOTHESIS - requires diagnostics", "diagnostics": r["diagnostics"],
                             "components": r["components"], "prevent": r["prevent"]})
        hyps.sort(key=lambda h: -h["likelihood"])
        ctx.extras["hypotheses"] = hyps
        if not hyps:
            return "No known fault pattern matched. Ask for symptoms, logs and the failure pattern."
        return json.dumps([{k: h[k] for k in ("hypothesis", "likelihood", "status", "diagnostics", "components",
                                              "prevent")} for h in hyps[:4]], indent=1)

    def rank_components(ctx, router_model: str = "", categories: str = "", symptoms: str = ""):
        cats = [c.lower() for c in _as_list(categories)]
        cats = [c for c in cats if c in COMPONENT_CATEGORIES] or \
            [c for h in ctx.extras.get("hypotheses", [])[:2] for c in h["components"]]
        if not cats:
            return ("Specify categories from: " + ", ".join(COMPONENT_CATEGORIES) +
                    " (or call diagnose_symptoms first).")
        model = agent.detect_model(router_model or symptoms)
        fam = []
        if model:
            digits = re.sub(r"\D", "", model[0])
            fam = [model[0].lower(), model[1].lower(), digits[:3]]
        qtok = set(tokenize(f"{router_model} {symptoms} {' '.join(cats)}"))
        recs = []
        for e in agent.component_catalog():
            if not set(e["categories"]) & set(cats):
                continue
            ctx_txt = " ".join(e["contexts"]).lower()
            compat = 1.0 if fam and any(f and f in ctx_txt + " " + e["pid"].lower() for f in fam) else 0.0
            if fam and not compat:
                continue
            sim = len(qtok & set(tokenize(ctx_txt))) / max(len(qtok), 1)
            score = round(0.5 + 0.3 * compat + 0.2 * sim, 3)
            recs.append({"title": e["pid"], "type": "component", "score": score,
                         "category": ", ".join(e["categories"]),
                         "rationale": f"{', '.join(e['categories'])}; "
                                      f"{'family match in datasheet' if compat else 'compatibility not verified'}",
                         "evidence": "; ".join(e["sources"]), "kpi": "repeat-incident rate / MTBF after replacement",
                         "datasheet_context": e["contexts"][0][:200]})
        recs.sort(key=lambda r: -r["score"])
        recs = recs[:5]
        if not recs:
            recs = [{"title": f"{c.title()} for {model[1] if model else 'the router'} (PID to confirm with datasheet/TAC)",
                     "type": "component category", "score": 0.5, "rationale": "no matching PID in downloaded datasheets",
                     "kpi": "repeat-incident rate"} for c in cats]
        ctx.add_recs(recs)
        return json.dumps([{k: r.get(k) for k in ("title", "category", "score", "rationale", "evidence",
                                                  "datasheet_context")} for r in recs], indent=1)

    def recommend_platforms(ctx, requirements: str):
        hits = Retriever("technology").search(agent.expand(requirements), k=8, boost={"company": ["Cisco"]})
        fam = defaultdict(lambda: {"score": 0.0, "cids": []})
        for h in hits:
            cid = ctx.add(h)
            name = re.sub(r"\s*\((catalog entry)\)|\s*datasheet", "", h.doc["source"]).strip()
            fam[name]["score"] += h.score
            fam[name]["cids"].append(cid)
        tot = sum(v["score"] for v in fam.values()) or 1
        recs = [{"title": n, "type": "platform", "score": round(v["score"] / tot, 3),
                 "rationale": f"share of requirement-matching evidence ({len(v['cids'])} passages)",
                 "evidence": ", ".join(v["cids"]), "kpi": "requirement fit, 3-5 yr TCO"}
                for n, v in sorted(fam.items(), key=lambda x: -x[1]["score"])][:4]
        ctx.add_recs(recs)
        return json.dumps(recs, indent=1) + "\n" + _fmt_hits(ctx, hits[:4], 350)

    return [
        Tool("search_tech_docs", "Search Cisco router datasheets/product pages and Google Cloud (MCP, RAG) docs.",
             {"query": "keywords", "company": "optional: Cisco | Google"}, search_tech_docs),
        Tool("identify_router_model", "Extract the Cisco model/family from text.", {"text": "client text"},
             identify_router_model),
        Tool("diagnose_symptoms", "Rule-based fault analysis: ranked hypotheses + diagnostics needed to confirm each.",
             {"symptoms": "symptom description"}, diagnose_symptoms),
        Tool("rank_components", "Rank compatible replacement parts (PIDs found in datasheets) for fault categories.",
             {"router_model": "e.g. ISR4331", "categories": "comma list: " + ", ".join(COMPONENT_CATEGORIES),
              "symptoms": "optional"}, rank_components),
        Tool("recommend_platforms", "Rank router platforms/solutions against requirements (evidence-weighted).",
             {"requirements": "throughput, sites, SD-WAN, cloud ..."}, recommend_platforms),
    ]


# ============================================================================ clinical
def clinical_tools() -> List[Tool]:
    from core.agents import TOPICS, ClinicalAgent
    agent = ClinicalAgent(None)

    def search_clinical_literature(ctx, query: str):
        hits = Retriever("clinical").search(agent.expand(query), k=5)
        return _fmt_hits(ctx, hits)

    def recommend_agilent_products(ctx, requirements: str):
        hits = Retriever("clinical").search(agent.expand(requirements), k=10, filters={"company": ["Agilent"]},
                                            per_source=1)
        prods, out = {}, []
        for h in hits:
            if h.doc.get("company") != "Agilent":
                continue
            cid = ctx.add(h)
            name = h.doc["source"].replace(" (catalog entry)", "")
            if name not in prods:
                prods[name] = {"title": name, "type": "product", "score": round(h.score, 2), "evidence": cid,
                               "rationale": " ".join(h.doc["text"].split())[:160], "url": h.doc.get("url", ""),
                               "kpi": "turnaround time, samples/day, cost per test"}
        recs = list(prods.values())[:4]
        ctx.add_recs(recs)
        if not recs:
            return "No Agilent product matched; ask for more specific lab requirements."
        return json.dumps(recs, indent=1)

    def search_clinical_trials(ctx, query: str, topic: str = "", status: str = ""):
        t = topic if topic in TOPICS else agent.detect_topic(query + " " + topic)
        filt = {"topic": [t]} if t else None
        boost = {"status": [s.strip().upper() for s in _as_list(status)]} if status else None
        hits = Retriever("trials").search(agent.expand(query), k=5, filters=filt, boost=boost, per_source=1)
        rows = []
        for h in hits:
            cid = ctx.add(h)
            d = h.doc
            rows.append({"cid": cid, "nct": d["nct"], "title": d["source"].split(": ", 1)[-1][:120],
                         "status": d["status"], "phase": d["phase"], "enrollment": d["enrollment"],
                         "sponsor": d["sponsor"], "interventions": d["interventions"][:120], "url": d["url"]})
        ctx.extras.setdefault("trials", []).extend(rows)
        ctx.add_recs([{"title": f"{r['nct']}: {r['title'][:80]}", "type": "clinical study", "score": 0,
                       "rationale": f"{r['status']}, phase {r['phase'] or 'n/a'}, n={r['enrollment']}",
                       "evidence": r["cid"], "kpi": "evidence quality"} for r in rows[:3]])
        return json.dumps(rows, indent=1) if rows else "No trials found."

    def trial_landscape(ctx, topic: str):
        t = topic if topic in TOPICS else agent.detect_topic(topic)
        if not t:
            return f"Unknown topic. Choose one of: {list(TOPICS)}"
        docs = [d for d in Retriever("trials").index.docs if d.get("topic") == t]
        land = {"topic": t, "studies": len(docs),
                "by_status": dict(Counter(d["status"] for d in docs).most_common(6)),
                "by_phase": dict(Counter(d["phase"] or "n/a" for d in docs).most_common(6)),
                "intervention_types": dict(Counter(m.group(1) for d in docs for m in re.finditer(
                    r"(DRUG|DEVICE|BEHAVIORAL|PROCEDURE|OTHER|DIAGNOSTIC_TEST|DIETARY_SUPPLEMENT|BIOLOGICAL)",
                    d["interventions"])).most_common(6))}
        ctx.extras["trial_landscape"] = land
        return json.dumps(land)

    return [
        Tool("search_clinical_literature", "Search clinical research PDFs and Agilent product pages.",
             {"query": "keywords"}, search_clinical_literature),
        Tool("recommend_agilent_products", "Rank Agilent diagnostic/lab products against lab requirements.",
             {"requirements": "lab needs"}, recommend_agilent_products),
        Tool("search_clinical_trials", "Find relevant ClinicalTrials.gov studies (32k records).",
             {"query": "condition/intervention", "topic": f"optional: {', '.join(TOPICS)}",
              "status": "optional: RECRUITING, COMPLETED ..."}, search_clinical_trials),
        Tool("trial_landscape", "Aggregate statistics of trials for a topic (status, phase, intervention types).",
             {"topic": ", ".join(TOPICS)}, trial_landscape),
    ]


# ============================================================================ SEC
def sec_tools() -> List[Tool]:
    from core.agents import COMPANIES, RISK_TAXONOMY, SECAgent
    agent = SECAgent(None)

    def list_filings(ctx):
        docs = Retriever("sec").index.docs
        f = sorted({(d["company"], d["year"], d["source"]) for d in docs})
        return "\n".join(f"- {c} | fiscal year {y} | {s}" for c, y, s in f)

    def search_filings(ctx, query: str, company: str = "", fiscal_year: str = ""):
        filt = {}
        comp = next((c for c, ks in COMPANIES.items() if company and any(k in company.lower() for k in ks)), "")
        if comp:
            filt["company"] = [comp]
        if fiscal_year:
            filt["year"] = [str(fiscal_year)]
        boost = None
        if comp and not fiscal_year:
            latest = max((d["year"] for d in Retriever("sec").index.docs if d["company"] == comp), default="")
            boost = {"year": [latest]}
        hits = Retriever("sec").search(agent.expand(query), k=5, filters=filt or None, boost=boost)
        return _fmt_hits(ctx, hits)

    def risk_matrix(ctx, companies: str, focus: str = "risk factors"):
        comps = [c for c, ks in COMPANIES.items() if any(k in companies.lower() for k in ks)] or ["Cisco", "Agilent"]
        matrix = {}
        for c in comps:
            hits = Retriever("sec").search(agent.expand(focus + " risk factors could adversely affect"), k=8,
                                           filters={"company": [c]})
            row = defaultdict(lambda: [0, []])
            for h in hits:
                cid = ctx.add(h)
                tl = " " + h.doc["text"].lower() + " "
                for cat, kws in RISK_TAXONOMY.items():
                    n = sum(tl.count(k) for k in kws)
                    if n:
                        row[cat][0] += n
                        row[cat][1].append(cid)
            matrix[c] = {k: {"mentions": v[0], "evidence": sorted(set(v[1]))} for k, v in
                         sorted(row.items(), key=lambda x: -x[1][0])}
        ctx.extras["risk_matrix"] = {c: {k: v["mentions"] for k, v in m.items()} for c, m in matrix.items()}
        ctx.add_recs([{"title": f"{c}: {k} risk", "type": "risk", "score": v["mentions"],
                       "rationale": f"{v['mentions']} risk-language mentions", "evidence": ", ".join(v["evidence"][:4])}
                      for c, m in matrix.items() for k, v in list(m.items())[:3]])
        return json.dumps(matrix, indent=1)

    def compare_companies(ctx, query: str, companies: str):
        comps = [c for c, ks in COMPANIES.items() if any(k in companies.lower() for k in ks)]
        if len(comps) < 2:
            return "Give at least two of: Cisco, Agilent, Alphabet."
        out = []
        for c in comps:
            out.append(f"## {c}\n" + search_filings(ctx, query, c))
        return "\n".join(out)

    return [
        Tool("list_filings", "List available 10-K filings (company, fiscal year).", {}, list_filings),
        Tool("search_filings", "Search 10-K text; filter by company and fiscal year (defaults to latest filing).",
             {"query": "keywords", "company": "optional: Cisco | Agilent | Alphabet",
              "fiscal_year": "optional, e.g. 2025"}, search_filings),
        Tool("risk_matrix", "Quantify disclosed risk categories per company from 10-K risk language.",
             {"companies": "e.g. Cisco, Agilent", "focus": "optional theme"}, risk_matrix),
        Tool("compare_companies", "Side-by-side evidence for the same question across companies.",
             {"query": "keywords", "companies": "e.g. Cisco, Agilent"}, compare_companies),
    ]


# ============================================================================ CSR
def csr_tools() -> List[Tool]:
    from core.agents import CSRAgent
    agent = CSRAgent(None)

    def search_sofi(ctx, query: str, report_year: str = ""):
        filt = {"year": [str(report_year)]} if str(report_year) in ("2023", "2024") else None
        hits = Retriever("csr").search(agent.expand(query), k=5, filters=filt)
        return _fmt_hits(ctx, hits)

    def rank_investments(ctx, goal: str):
        spec = agent.specialist("Which investment should we fund to " + goal, [])
        recs = spec["recommendations"]
        ctx.add_recs([{k: v for k, v in r.items() if k != "snippet"} for r in recs])
        ctx.missing_info += [m for m in spec.get("missing_info", []) if m not in ctx.missing_info]
        return json.dumps([{k: r[k] for k in ("title", "score", "rationale", "evidence", "kpi")} for r in recs],
                          indent=1)

    return [
        Tool("search_sofi", "Search FAO 'State of Food Security and Nutrition in the World' reports.",
             {"query": "keywords", "report_year": "optional: 2023 | 2024 (edition)"}, search_sofi),
        Tool("rank_investments", "Rank CSR/ESG investment options by goal relevance and SOFI evidence strength.",
             {"goal": "what the client wants to achieve"}, rank_investments),
    ]


# ============================================================================ retail
def retail_tools() -> List[Tool]:
    from core.agents import RetailAgent
    from core.basket import exact_cart_seen
    agent = RetailAgent(None)

    def identify_cart_items(ctx, text: str):
        items = agent.parse_cart(text)
        ctx.extras["cart"] = items
        return json.dumps(items) if items else ("No catalogue item matched. Catalogue examples: " +
                                                "; ".join(agent.model().items[:8]))

    def recommend_addons(ctx, items: str):
        m = agent.model()
        cart = [i for i in m.items if any(x and x.lower() in i.lower() for x in _as_list(items))] or \
            agent.parse_cart(str(items))
        recs = m.recommend(cart, k=5)
        ctx.extras.update({"cart": cart, "recs": [r.to_dict() for r in recs],
                           "exact_cart_seen_in_history": exact_cart_seen(m.tx, cart)})
        out = []
        for r in recs:
            h = Hit(doc={"source": f"Transaction history ({m.n} baskets)", "path": "retail", "page": "",
                         "chunk": r.item, "doc_type": "catalog",
                         "text": f"Rule {{{', '.join(r.antecedent)}}} -> {r.item}: support {r.support_count}, "
                                 f"confidence {r.confidence:.3f}, lift {r.lift:.2f}"}, score=r.score)
            cid = ctx.add(h)
            out.append({"cid": cid, **r.to_dict()})
        ctx.add_recs([{"title": r.item, "type": "cross-sell", "score": round(r.score, 3),
                       "rationale": f"confidence {r.confidence:.0%}, lift {r.lift:.2f}", "evidence": o["cid"],
                       "kpi": "attach rate, AOV"} for r, o in zip(recs, out)])
        if not cart:
            return "Cart items not recognised."
        return json.dumps(out, indent=1) if out else "No reliable rule (support/lift thresholds) - recommend nothing."

    def top_associations(ctx, k: int = 8):
        return json.dumps(agent.model().top_rules(int(k)), indent=1)

    return [
        Tool("identify_cart_items", "Map product names/SKUs in text to catalogue items.", {"text": "text"},
             identify_cart_items),
        Tool("recommend_addons", "Association-rule add-on recommendations for a cart (abstains if no pattern).",
             {"items": "comma-separated item names or SKUs"}, recommend_addons),
        Tool("top_associations", "Strongest co-purchase rules overall.", {"k": "number"}, top_associations),
    ]


TOOLSETS = {"technology": technology_tools, "clinical": clinical_tools, "sec": sec_tools, "csr": csr_tools,
            "retail": retail_tools}
_cache: Dict[str, List[Tool]] = {}


def get_tools(domain: str) -> Dict[str, Tool]:
    if domain not in _cache:
        _cache[domain] = TOOLSETS[domain]()
    return {t.name: t for t in _cache[domain]}
