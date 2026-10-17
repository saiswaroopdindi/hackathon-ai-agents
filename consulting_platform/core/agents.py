"""Specialist domain agents.

Every agent follows the same loop:
  1. query understanding  (entity / year / model detection, domain query expansion)
  2. retrieval            (hybrid BM25 over its own knowledge base, metadata filters)
  3. specialist reasoning (a domain recommender: fault hypotheses + component ranking, product/trial
                           ranking, risk matrix, investment ranking, market-basket rules)
  4. synthesis            (LLM grounded on numbered evidence; extractive fallback without an LLM)
  5. self-assessment      (confidence + missing information)
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from core.llm import LLM, LLMError
from core.retriever import Hit, Retriever
from core.text import split_sentences, tokenize


# =============================================================================== data model
@dataclass
class AgentResult:
    domain: str
    label: str
    question: str
    answer: str = ""
    evidence: List[Dict] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)
    next_actions: List[Dict] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    confidence: Dict = field(default_factory=dict)
    extras: Dict = field(default_factory=dict)
    mode: str = "extractive"
    latency_s: float = 0.0
    error: str = ""
    task: str = ""                      # task given by the Manager (may differ from the client question)
    steps: List[Dict] = field(default_factory=list)   # reasoning trace (thought / action / observation)
    tool_calls: int = 0
    tool_errors: int = 0
    revisions: int = 0
    self_check: Dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__.copy()


SYSTEM_TEMPLATE = """You are the {label} of an AI management-consulting firm. Knowledge base: {sources}.
Rules:
- Use ONLY the numbered evidence provided. Cite every factual claim inline like [{p}1] or [{p}2][{p}4].
- If the evidence does not support an answer, say so plainly and state what information is needed.
- Separate CONFIRMED facts (stated in evidence or by the customer) from HYPOTHESES (your inference).
- Be specific, concise and business-oriented: what to do next, why, and how to measure impact.
- Never invent part numbers, figures, product names or URLs that are not in the evidence or recommender output.
{extra}"""

ANSWER_FORMAT = """Write the answer in Markdown with these sections:
### Answer
(3-6 sentences directly answering the question, with citations)
### Key evidence
(3-5 bullets, each with a citation)
### Recommendations
(ranked numbered list; each item: **action** - why (cite) - how to measure (KPI))
### Gaps & assumptions
(bullets: what is uncertain or what the customer must provide)"""


def _fmt_evidence(hits: List[Hit], max_chars: int = 1100) -> str:
    out = []
    for h in hits:
        out.append(f"[{h.cid}] ({h.citation})\n{h.doc['text'][:max_chars]}")
    return "\n\n".join(out)


# =============================================================================== base agent
class BaseAgent:
    domain = "base"
    prefix = "S"
    index_name = ""
    expansions: Dict[str, str] = {}
    extra_rules = ""

    def __init__(self, llm: Optional[LLM] = None, top_k: int = config.TOP_K):
        self.llm = llm
        self.top_k = top_k
        meta = config.DOMAINS.get(self.domain, {})
        self.label = meta.get("label", self.domain)
        self.sources = meta.get("sources", "")

    # ---------------------------------------------------------------- hooks
    def expand(self, q: str) -> str:
        ql = q.lower()
        add = [v for k, v in self.expansions.items() if k in ql]
        return q + (" " + " ".join(add) if add else "")

    def filters(self, q: str) -> Dict:
        return {}

    def retrieve(self, q: str) -> List[Hit]:
        f = self.filters(q)
        hits = Retriever(self.index_name).search(self.expand(q), k=self.top_k, filters=f.get("filters"),
                                                 boost=f.get("boost"))
        return hits

    def specialist(self, q: str, hits: List[Hit]) -> Dict:
        return {}

    # ---------------------------------------------------------------- main
    def run(self, q: str, sub_question: Optional[str] = None) -> AgentResult:
        t0 = time.time()
        question = sub_question or q
        res = AgentResult(domain=self.domain, label=self.label, question=question)
        try:
            hits = self.retrieve(question)
            for i, h in enumerate(hits, 1):
                h.cid = f"{self.prefix}{i}"
            spec = self.specialist(question, hits)
            res.evidence = [h.to_dict() for h in hits]
            res.recommendations = spec.get("recommendations", [])
            res.next_actions = spec.get("next_actions", [])
            res.missing_info = spec.get("missing_info", [])
            res.extras = spec.get("extras", {})
            res.confidence = self.assess(question, hits, spec)
            res.answer, res.mode = self.synthesize(question, hits, spec)
        except Exception as e:  # never let one agent crash the orchestration
            res.error = f"{type(e).__name__}: {e}"
            res.answer = f"_{self.label} could not complete: {res.error}_"
            res.confidence = {"score": 0.0, "level": "Low", "reasons": ["agent error"]}
        res.latency_s = round(time.time() - t0, 2)
        return res

    # ---------------------------------------------------------------- confidence
    def assess(self, q: str, hits: List[Hit], spec: Dict) -> Dict:
        if not hits:
            return {"score": 0.05, "level": "Low", "reasons": ["no supporting evidence retrieved"]}
        qt = set(tokenize(q))
        top_text = set(tokenize(" ".join(h.doc["text"] for h in hits[:3])))
        coverage = len(qt & top_text) / max(len(qt), 1)
        top = hits[0].score
        strength = top / (top + 12.0) if top > 1.5 else top            # BM25 -> (0,1); dense scores already ~[0,1]
        strong = [h for h in hits if h.score >= 0.6 * top]
        agreement = min(len({h.doc.get("path", "") + str(h.doc.get("page", "")) for h in strong}) / 3, 1.0)
        score = 0.45 * coverage + 0.35 * strength + 0.20 * agreement
        reasons = [f"query-term coverage in top evidence {coverage:.0%}",
                   f"retrieval strength {strength:.2f}", f"{len(strong)} corroborating passages"]
        cap = spec.get("confidence_cap")
        if cap is not None and score > cap:
            score = cap
            reasons.append(spec.get("cap_reason", "capped by specialist"))
        level = "High" if score >= 0.7 else "Medium" if score >= 0.48 else "Low"
        return {"score": round(score, 2), "level": level, "reasons": reasons}

    # ---------------------------------------------------------------- synthesis
    def specialist_brief(self, spec: Dict) -> str:
        recs = spec.get("recommendations", [])
        if not recs:
            return ""
        lines = ["Recommender output (ranked, computed by the platform's recommendation engine):"]
        for i, r in enumerate(recs[:6], 1):
            lines.append(f"{i}. {r.get('title')} | score={r.get('score')} | {r.get('rationale','')}"
                         f"{' | evidence ' + r['evidence'] if r.get('evidence') else ''}")
        if spec.get("missing_info"):
            lines.append("Missing information flagged: " + "; ".join(spec["missing_info"]))
        if spec.get("brief"):
            lines.append(spec["brief"])
        return "\n".join(lines)

    def synthesize(self, q: str, hits: List[Hit], spec: Dict):
        if self.llm is not None and self.llm.available and hits:
            system = SYSTEM_TEMPLATE.format(label=self.label, sources=self.sources, p=self.prefix,
                                            extra=self.extra_rules)
            prompt = (f"Customer question: {q}\n\nEvidence:\n{_fmt_evidence(hits)}\n\n"
                      f"{self.specialist_brief(spec)}\n\n{ANSWER_FORMAT}")
            try:
                return self.llm.generate(prompt, system=system), "llm"
            except LLMError as e:
                spec.setdefault("missing_info", []).append(f"(LLM unavailable, extractive answer used: {str(e)[:80]})")
        return self.extractive(q, hits, spec), "extractive"

    def extractive(self, q: str, hits: List[Hit], spec: Dict) -> str:
        qt = set(tokenize(self.expand(q)))
        cands = []
        for h in hits[:6]:
            for s in split_sentences(h.doc["text"]):
                st = tokenize(s)
                if not st or s.startswith(("Source:", "http")) or len(st) < 5:
                    continue
                digits = sum(c.isdigit() for c in s) / len(s)
                if digits > 0.15:
                    continue
                ov = len(qt & set(st))
                cands.append((ov + 0.15 * (7 - h.rank) + min(len(st), 25) / 50, s, h.cid))
        cands.sort(key=lambda x: -x[0])
        chosen, used = [], Counter()
        for sc, s, cid in cands:
            if used[cid] >= 2 or any(s[:60] == c[0][:60] for c in chosen):
                continue
            chosen.append((s, cid))
            used[cid] += 1
            if len(chosen) >= 5:
                break
        out = ["### Answer (extractive - evidence sentences ranked by relevance)"]
        out += [f"- {s} [{cid}]" for s, cid in chosen] or ["- No relevant evidence found in the knowledge base."]
        recs = spec.get("recommendations", [])
        if recs:
            out.append("\n### Recommendations")
            for i, r in enumerate(recs[:5], 1):
                ev = f" [{r['evidence']}]" if r.get("evidence") else ""
                out.append(f"{i}. **{r['title']}** - {r.get('rationale','')}{ev}"
                           + (f" - *KPI:* {r['kpi']}" if r.get("kpi") else ""))
        if spec.get("missing_info"):
            out.append("\n### Gaps & assumptions")
            out += [f"- {m}" for m in spec["missing_info"]]
        return "\n".join(out)


# =============================================================================== Technology
FAULT_RULES = [
    {"name": "Physical layer: interface, optic (SFP) or cabling fault",
     "triggers": ["intermittent", "flap", "crc", "input error", "link down", "connectivity", "drops", "dropping",
                  "packet loss", "interface", "port", "sfp", "fiber", "cable"],
     "diagnostics": ["show interfaces <intf>  - CRC/input errors, interface resets, last flap time",
                     "show logging | include LINK|LINEPROTO  - correlate flaps with timestamps",
                     "show interfaces transceiver detail  - optical Rx/Tx power for SFP ports (if fiber)",
                     "swap cable / move to spare port to isolate port vs. optic vs. cable"],
     "components": ["sfp", "interface module"],
     "prevent": "Track interface error counters and optic light levels in monitoring; keep spare optics on site."},
    {"name": "Power supply or power-quality problem",
     "triggers": ["reboot", "reload", "restart", "power", "shut down", "shutdown", "brownout", "ups", "random"],
     "diagnostics": ["show version  - 'Last reload reason' and uptime",
                     "show environment all  - PSU status, voltage, alarms",
                     "check UPS / PDU logs for the same timestamps"],
     "components": ["power supply"],
     "prevent": "Redundant/dual PSU where the chassis supports it, UPS-backed circuits, environmental alarms."},
    {"name": "Thermal / fan failure",
     "triggers": ["overheat", "temperature", "hot", "fan", "thermal", "noise", "heat"],
     "diagnostics": ["show environment all  - temperature sensors and fan status",
                     "inspect airflow, dust, rack placement"],
     "components": ["fan"],
     "prevent": "Temperature alerting via SNMP/telemetry, airflow audits, scheduled cleaning."},
    {"name": "Software defect, CPU or memory exhaustion",
     "triggers": ["crash", "slow", "high cpu", "cpu", "memory", "leak", "traceback", "hang", "freeze", "bug",
                  "after upgrade", "software", "ios", "firmware"],
     "diagnostics": ["show processes cpu sorted / show processes memory sorted",
                     "show platform resources",
                     "show version  - IOS XE release; check Cisco Bug Search for known defects",
                     "collect crashinfo / core files if a crash occurred"],
     "components": ["memory", "license"],
     "prevent": "Standardise on a recommended IOS XE release, staged upgrades, CPU/memory trend alerts."},
    {"name": "WAN / routing / VPN control-plane instability (not hardware)",
     "triggers": ["bgp", "ospf", "isp", "vpn", "tunnel", "ipsec", "latency", "routing", "sd-wan", "wan", "jitter"],
     "diagnostics": ["show ip bgp summary / show ip ospf neighbor  - neighbor flaps",
                     "show crypto session  - VPN/IPsec tunnel stability",
                     "IP SLA / continuous ping + traceroute to isolate ISP vs. LAN side"],
     "components": [],
     "prevent": "Dual-ISP / SD-WAN path redundancy with application-aware routing; IP SLA tracking."},
    {"name": "Capacity / performance ceiling of the platform",
     "triggers": ["throughput", "bandwidth", "congestion", "slow", "performance", "capacity", "growth", "saturat"],
     "diagnostics": ["show platform hardware qfp active datapath utilization  - forwarding load",
                     "interface utilisation trends (peak vs. licensed throughput)"],
     "components": ["license", "platform"],
     "prevent": "Capacity planning against datasheet throughput; performance licence or platform refresh."},
]

COMPONENT_CATEGORIES = {
    "power supply": ["power supply", "power supplies", "psu", "pwr-", "watt", " ac ", " dc "],
    "fan": ["fan", "airflow", "cooling"],
    "sfp": ["sfp", "transceiver", "optic", "glc-", "fiber"],
    "interface module": ["nim", "sm-x", "module", "network interface", "epa", "spa"],
    "memory": ["dram", "memory", "flash", "mem-", "ssd", "storage"],
    "license": ["license", "licence", "fl-", "performance", "boost", "dna"],
    "platform": ["chassis", "router", "platform", "/k9", "c8300", "c8200", "isr4", "asr1"],
}

MODEL_PATTERNS = [
    (r"isr\s?-?(4\d{3})", lambda m: (f"ISR{m.group(1)}", "Cisco 4000 Series ISR")),
    (r"\b(4\d{3})\s?(?:series)?\s?(?:isr|router)", lambda m: (f"ISR{m.group(1)}", "Cisco 4000 Series ISR")),
    (r"(?:catalyst|c)\s?-?8300", lambda m: ("C8300", "Catalyst 8300 Series")),
    (r"(?:catalyst|c)\s?-?8200", lambda m: ("C8200", "Catalyst 8200 Series")),
    (r"8000v", lambda m: ("C8000V", "Catalyst 8000V Edge Software")),
    (r"asr\s?-?(1\d{3})", lambda m: (f"ASR{m.group(1)}", "ASR 1000 Series")),
]

PID_RE = re.compile(r"\b((?:NIM|SM-X|SM|PWR|FL|C8\d{3}|C8\d{2}|ISR4\d{3}|ASR1\d{2,3}|SFP|GLC|MEM|ASR1000|"
                    r"C-NIM|ACS|CON|SSD|EHWIC|P-LTE|NIM-LTEA|C1100|CAB|A900|PVDM|ESP|RP\d|EPA|SPA)"
                    r"[A-Z0-9]*(?:-[A-Z0-9\.]+){0,6}(?:/K9)?=?)")


class TechnologyAgent(BaseAgent):
    domain, prefix, index_name = "technology", "T", "technology"
    expansions = {"router": "router platform interfaces", "outage": "redundancy availability",
                  "replace": "replacement module power supply", "part": "module power supply fan",
                  "mcp": "model context protocol", "rag": "retrieval augmented generation grounding"}
    extra_rules = ("- For hardware faults: never conclude a component has failed without diagnostic evidence; "
                   "list the diagnostics required to confirm each hypothesis before ordering parts.\n"
                   "- Recommend part numbers only if they appear in the evidence/recommender output.")
    _catalog = None

    def detect_model(self, q: str):
        ql = q.lower()
        for pat, fn in MODEL_PATTERNS:
            m = re.search(pat, ql)
            if m:
                return fn(m)
        return None

    def filters(self, q):
        m = self.detect_model(q)
        return {"boost": {"company": ["Cisco"]}} if (m or "cisco" in q.lower()) else {}

    PREFIX_CATEGORY = [("PWR", "power supply"), ("FL-", "license"), ("SFP", "sfp"), ("GLC", "sfp"),
                       ("MEM", "memory"), ("SSD", "memory"), ("NIM", "interface module"), ("SM-X", "interface module"),
                       ("EHWIC", "interface module"), ("EPA", "interface module"), ("SPA", "interface module"),
                       ("PVDM", "interface module"), ("ACS", "accessory"), ("CAB", "cable"),
                       ("ISR", "platform"), ("C8", "platform"), ("ASR", "platform"), ("C1100", "platform")]

    @classmethod
    def categorize(cls, pid: str, ctx: str) -> List[str]:
        """Category from the PID prefix (reliable); fall back to the words right next to the PID."""
        cats = [c for p, c in cls.PREFIX_CATEGORY if pid.upper().startswith(p)]
        if "SFP" in pid.upper() and "sfp" not in cats:
            cats.append("sfp")
        if not cats:
            ctx_l = ctx.lower()
            cats = [c for c, kws in COMPONENT_CATEGORIES.items() if any(k in ctx_l for k in kws)][:1]
        return cats

    @classmethod
    def component_catalog(cls) -> List[Dict]:
        """Extract Cisco product IDs (PIDs) + context from the downloaded datasheets."""
        if cls._catalog is not None:
            return cls._catalog
        cat = {}
        try:
            docs = Retriever("technology").index.docs
        except FileNotFoundError:
            docs = []
        for d in docs:
            if d.get("doc_type") not in ("datasheet", "web"):
                continue
            text = d["text"]
            for m in PID_RE.finditer(text):
                pid = m.group(1).rstrip(".")
                if len(pid) < 5 or "-" not in pid and not re.search(r"\d", pid):
                    continue
                ctx = text[max(0, m.start() - 160): m.end() + 220].replace("\n", " ")
                cats = cls.categorize(pid, text[max(0, m.start() - 60): m.end() + 90])
                e = cat.setdefault(pid, {"pid": pid, "contexts": [], "categories": Counter(), "sources": set()})
                if len(e["contexts"]) < 3:
                    e["contexts"].append(ctx.strip())
                e["categories"].update(cats)
                e["sources"].add(f"{d['source']}, p.{d.get('page', 1)}")
        cls._catalog = [{"pid": e["pid"], "contexts": e["contexts"],
                         "categories": [c for c, _ in e["categories"].most_common(2)],
                         "sources": sorted(e["sources"])[:2]} for e in cat.values()]
        return cls._catalog

    def specialist(self, q, hits):
        ql = q.lower()
        model = self.detect_model(q)
        family_terms = []
        if model:
            pid_base, family = model
            digits = re.sub(r"\D", "", pid_base)
            family_terms = [pid_base.lower(), family.lower(), digits[:3]]
        is_fault = any(t in ql for r in FAULT_RULES for t in r["triggers"][:8]) or \
            any(w in ql for w in ("fail", "issue", "problem", "outage", "error"))
        spec = {"recommendations": [], "next_actions": [], "missing_info": [], "extras": {}}

        # ---- 1. fault hypotheses (rule-based diagnosis) -------------------------------------
        hyps = []
        if is_fault:
            for r in FAULT_RULES:
                matched = [t for t in r["triggers"] if t in ql]
                if matched:
                    hyps.append({"hypothesis": r["name"], "matched_symptoms": matched,
                                 "likelihood": round(min(0.2 + 0.18 * len(matched), 0.85), 2),
                                 "status": "HYPOTHESIS - requires diagnostics", "diagnostics": r["diagnostics"],
                                 "components": r["components"], "prevent": r["prevent"]})
            hyps.sort(key=lambda h: -h["likelihood"])
            confirmed_words = ("crc errors", "show environment", "psu failed", "fan failed", "alarm", "we observed",
                               "logs show", "diagnostics show")
            if any(c in ql for c in confirmed_words):
                for h in hyps[:1]:
                    h["status"] = "SUPPORTED by customer-reported diagnostics"
            spec["extras"]["hypotheses"] = hyps

        # ---- 2. component recommender (similarity + compatibility ranking) ------------------
        wanted = Counter()
        for h in hyps[:3]:
            for c in h["components"]:
                wanted[c] += h["likelihood"]
        for c, kws in COMPONENT_CATEGORIES.items():
            if c != "platform" and any(k.strip() in ql for k in kws if len(k.strip()) > 3):
                wanted[c] += 0.5
        if any(w in ql for w in ("refresh", "migrate", "new router", "replace the router", "replace router",
                                 "replacement router", "end of life", "end-of-life")):
            wanted["platform"] += 0.6
        qtok = set(tokenize(q))
        recs = []
        for e in self.component_catalog():
            ctx = " ".join(e["contexts"]).lower()
            cat_score = max([wanted.get(c, 0) for c in e["categories"]] or [0])
            if wanted and cat_score == 0:
                continue
            compat = 0.0
            if family_terms:
                compat = 1.0 if any(ft and ft in (ctx + " " + e["pid"].lower()) for ft in family_terms) else 0.0
                if compat == 0:
                    continue
            sim = len(qtok & set(tokenize(ctx))) / max(len(qtok), 1)
            score = 0.5 * min(cat_score, 1) + 0.3 * compat + 0.2 * sim
            if score <= 0.15:
                continue
            recs.append({"title": e["pid"], "type": "component", "score": round(score, 3),
                         "category": ", ".join(e["categories"]),
                         "rationale": f"{', '.join(e['categories']) or 'component'}; "
                                      f"{'compatible family mentioned in datasheet' if compat else 'family not verified'}; "
                                      f"context: \"{e['contexts'][0][:160]}...\"",
                         "evidence": "; ".join(e["sources"]), "kpi": "MTBF / repeat-incident rate after replacement"})
        recs.sort(key=lambda r: -r["score"])
        recs = recs[:5]
        if not recs and wanted:
            # catalogue not downloaded or nothing compatible: recommend categories, not invented PIDs
            for c, w in wanted.most_common(3):
                recs.append({"title": f"{c.title()} (exact PID to be confirmed against the {model[1] if model else 'router'} datasheet / Cisco TAC)",
                             "type": "component category", "score": round(min(w, 1), 2),
                             "rationale": "implied by the leading fault hypotheses",
                             "kpi": "repeat-incident rate after replacement"})
        if not is_fault:
            # platform / solution recommender: aggregate evidence by product family (source document)
            fam = defaultdict(lambda: {"score": 0.0, "cids": []})
            for h in hits:
                name = re.sub(r"\s*\((catalog entry)\)|\s*datasheet", "", h.doc["source"]).strip()
                fam[name]["score"] += h.score
                fam[name]["cids"].append(h.cid)
            tot = sum(v["score"] for v in fam.values()) or 1
            recs = [{"title": n, "type": "platform / solution", "score": round(v["score"] / tot, 3),
                     "rationale": f"share of retrieved evidence matching the requirements ({len(v['cids'])} passages)",
                     "evidence": ", ".join(v["cids"]), "kpi": "requirement fit vs. datasheet (throughput, interfaces, TCO)"}
                    for n, v in sorted(fam.items(), key=lambda x: -x[1]["score"])][:4]
        spec["recommendations"] = recs

        # ---- 3. consulting next actions -----------------------------------------------------
        na = []
        if not is_fault:
            na += [{"action": "Document requirements (sites, throughput incl. IPsec, interfaces, SD-WAN/cloud, security)",
                    "why": "Platform choice must be checked against datasheet limits, not marketing names",
                    "kpi": "Requirements sign-off", "priority": "P1"},
                   {"action": "Shortlist the top-ranked platform(s) and validate with a pilot site",
                    "why": "Confirms performance with real traffic before fleet rollout",
                    "kpi": "Pilot throughput, latency, incident count", "priority": "P2"},
                   {"action": "Build a 3-5 year TCO incl. licences, support and refresh timing",
                    "why": "Licensing and support dominate lifetime cost", "kpi": "TCO per site", "priority": "P2"}]
        if hyps:
            na.append({"action": "Run the diagnostic checklist before ordering hardware",
                       "why": "Avoids replacing healthy parts; confirms which hypothesis is real",
                       "kpi": "Root cause confirmed within 48h", "priority": "P1"})
            na.append({"action": "Open a Cisco TAC case / RMA if a component fault is confirmed and the device is under contract",
                       "why": "Advance replacement is typically faster and cheaper than ad-hoc purchase",
                       "kpi": "Time-to-restore (MTTR)", "priority": "P1"})
            na.append({"action": "Order only the components confirmed by diagnostics (plus one critical spare)",
                       "why": "Procurement priority follows evidence, not guesses", "kpi": "Spare-parts spend vs. downtime avoided",
                       "priority": "P2"})
            na.append({"action": "Preventive: " + "; ".join(dict.fromkeys(h["prevent"] for h in hyps[:3])),
                       "why": "Reduce recurrence of the same failure class", "kpi": "Outage minutes per month, MTBF",
                       "priority": "P2"})
        if model and model[0].startswith("ISR4"):
            na.append({"action": "Review platform lifecycle: compare ISR 4000 with Catalyst 8200/8300 as a refresh path",
                       "why": "Refresh decision balances repair cost vs. support horizon (verify end-of-life dates with Cisco)",
                       "kpi": "3-year TCO, support coverage", "priority": "P3"})
        spec["next_actions"] = na

        # ---- 4. missing information -----------------------------------------------------------
        miss = []
        if is_fault:
            if not model:
                miss.append("Exact router model / PID (e.g. ISR4331/K9, C8300-1N1S-4T2X)")
            if not re.search(r"\b1[67]\.\d+", ql):
                miss.append("IOS XE software release (show version)")
            miss += ["Interface error counters and syslog around the failure times",
                     "Environmental status (show environment all) and last reload reason",
                     "Pattern of the failures (time of day, traffic level, after a change?)",
                     "Service contract status (SmartNet) for RMA eligibility"]
            spec["confidence_cap"] = 0.55
            spec["cap_reason"] = "root cause not confirmed by diagnostics -> confidence capped at Medium"
        spec["missing_info"] = miss
        if hyps:
            spec["brief"] = "Fault hypotheses (rule-based, ranked):\n" + "\n".join(
                f"- {h['hypothesis']} (likelihood {h['likelihood']}, {h['status']}); diagnostics: "
                f"{' | '.join(h['diagnostics'][:3])}" for h in hyps[:4])
        return spec


# =============================================================================== Clinical
TOPICS = {"diabetes": ["diabet", "glucose", "insulin", "glycemic", "hba1c"],
          "covid": ["covid", "sars-cov", "coronavirus", "long covid"],
          "KneeInjuries": ["knee", "acl", "arthroplasty", "meniscus", "exoskeleton"],
          "Hearattack": ["heart attack", "myocardial", "cardiac", "heart disease", "coronary"]}


class ClinicalAgent(BaseAgent):
    domain, prefix, index_name = "clinical", "C", "clinical"
    expansions = {"heart attack": "myocardial infarction cardiac", "lab": "laboratory",
                  "slide": "digital pathology scanner", "metabol": "seahorse cell metabolic analysis",
                  "stain": "special stains artisan"}
    extra_rules = ("- This is decision support, not medical advice; flag that clinical/regulatory validation "
                   "(e.g. IVD status, lab accreditation) is required before adoption.")

    def detect_topic(self, q):
        ql = q.lower()
        for t, kws in TOPICS.items():
            if any(k in ql for k in kws):
                return t
        return None

    def specialist(self, q, hits):
        ql = q.lower()
        spec = {"recommendations": [], "next_actions": [], "missing_info": [], "extras": {}}
        # ---- product recommender over the Agilent catalogue (content-based) -------------------
        wants_product = any(w in ql for w in ("instrument", "lab", "laborator", "recommend", "which", "equipment",
                                              "scanner", "pathology", "stain", "metabolic", "component", "agilent"))
        if wants_product:
            r = Retriever("clinical")
            prod_hits = r.search(self.expand(q), k=12, filters={"company": ["Agilent"]}, per_source=1)
            prods = {}
            for h in prod_hits:
                if h.doc.get("company") != "Agilent":
                    continue
                name = h.doc["source"].replace(" (catalog entry)", "")
                if name not in prods:
                    prods[name] = {"title": name, "type": "product", "score": round(h.score, 2),
                                   "rationale": h.doc["text"][:180].replace("\n", " "),
                                   "evidence": h.doc.get("url", ""),
                                   "kpi": "turnaround time, samples/day, cost per test"}
            spec["recommendations"] += list(prods.values())[:3]
            if prods:
                spec["missing_info"] += ["Sample volume per day and required turnaround time",
                                         "Regulatory need (IVD vs. research-use-only)", "Budget and LIS integration needs"]
            if any(w in ql for w in ("component", "fail", "error", "repair", "replace")):
                spec["missing_info"].append("Instrument model/serial number and error codes (troubleshooting request)")
        # ---- clinical-trial recommender (similarity + status-aware ranking) ----------------------
        topic = self.detect_topic(q)
        if not topic and not any(w in ql for w in ("trial", "study", "studies", "evidence", "patient")):
            spec["next_actions"] = self._actions()
            return spec
        tr = Retriever("trials")
        f = {"topic": [topic]} if topic else None
        boost = {"status": ["RECRUITING", "ACTIVE_NOT_RECRUITING"]} if any(
            w in ql for w in ("recruit", "ongoing", "active", "enrol", "current")) else {"status": ["COMPLETED"]}
        trials = tr.search(self.expand(q), k=5, filters=f, boost=boost, per_source=1)
        spec["extras"]["trials"] = [{"nct": h.doc["nct"], "title": h.doc["source"].split(": ", 1)[-1],
                                     "status": h.doc["status"], "phase": h.doc["phase"],
                                     "enrollment": h.doc["enrollment"], "sponsor": h.doc["sponsor"],
                                     "interventions": h.doc["interventions"][:150], "url": h.doc["url"],
                                     "score": round(h.score, 2)} for h in trials]
        if topic:
            docs = [d for d in tr.index.docs if d.get("topic") == topic]
            spec["extras"]["trial_landscape"] = {
                "topic": topic, "studies": len(docs),
                "by_status": dict(Counter(d["status"] for d in docs).most_common(5)),
                "top_intervention_types": dict(Counter(
                    m.group(1) for d in docs for m in re.finditer(r"(DRUG|DEVICE|BEHAVIORAL|PROCEDURE|OTHER|"
                                                                   r"DIAGNOSTIC_TEST|DIETARY_SUPPLEMENT|BIOLOGICAL)",
                                                                   d["interventions"])).most_common(5))}
        for t in spec["extras"]["trials"][:3]:
            spec["recommendations"].append({"title": f"{t['nct']}: {t['title'][:90]}", "type": "clinical study",
                                            "score": t["score"],
                                            "rationale": f"{t['status']}, phase {t['phase'] or 'n/a'}, n={t['enrollment']}, sponsor {t['sponsor']}",
                                            "evidence": t["url"], "kpi": "evidence quality (phase, enrollment, results posted)"})
        spec["next_actions"] = self._actions()
        return spec

    @staticmethod
    def _actions():
        return [
            {"action": "Shortlist the top-ranked option(s) and request vendor demos / validation data",
             "why": "Recommendation is based on documented capabilities and peer-reviewed evidence",
             "kpi": "Pilot accuracy / turnaround vs. current baseline", "priority": "P1"},
            {"action": "Run a 60-90 day pilot with predefined clinical and operational KPIs",
             "why": "Validates fit with your workflow before capital commitment",
             "kpi": "Throughput, error/re-run rate, cost per test", "priority": "P2"},
            {"action": "Confirm regulatory/IVD status and data-privacy (HIPAA) requirements",
             "why": "Clinical deployment requires compliance sign-off", "kpi": "Compliance checklist completed",
             "priority": "P1"}]


# =============================================================================== SEC
COMPANIES = {"Cisco": ["cisco"], "Agilent": ["agilent"], "Alphabet": ["alphabet", "google"]}
RISK_TAXONOMY = {
    "Competitive": ["compet", "market share", "pricing pressure", "new entrant", "rival"],
    "Supply chain & operations": ["supply chain", "component shortage", "manufactur", "supplier", "inventory",
                                  "logistic", "contract manufacturer", "operational"],
    "Macroeconomic & trade": ["tariff", "inflation", "interest rate", "macroeconomic", "currency", "foreign exchange",
                              "recession", "trade"],
    "Regulatory & legal": ["regulat", "litigation", "legal proceeding", "antitrust", "compliance", "government",
                           "privacy law", "lawsuit", "fda"],
    "Cybersecurity & IT": ["cyber", "security breach", "ransomware", "data breach", "vulnerabil"],
    "Financial": ["debt", "liquidity", "impairment", "goodwill", "credit", "cash flow", "margin", "revenue declin"],
    "Technology & AI transition": ["artificial intelligence", " ai ", "technological change", "innovation",
                                   "new products", "cloud"],
    "Acquisition & integration": ["acquisition", "integrat", "divestiture", "splunk", "merger"],
}


class SECAgent(BaseAgent):
    domain, prefix, index_name = "sec", "F", "sec"
    expansions = {"risk": "risk factors could adversely affect", "competit": "competition competitors",
                  "financial": "revenue gross margin net income cash", "operational": "operations supply chain manufacturing"}
    extra_rules = "- Quote figures exactly as filed and state the fiscal year they refer to."

    def companies(self, q):
        ql = q.lower()
        found = [c for c, ks in COMPANIES.items() if any(k in ql for k in ks)]
        if not found and any(w in ql for w in ("these companies", "both companies", "the companies")):
            found = ["Cisco", "Agilent"]
        return found

    def years(self, q):
        return re.findall(r"\b(20\d\d)\b", q)

    def retrieve(self, q):
        comps = self.companies(q) or [None]
        yrs = self.years(q)
        r = Retriever("sec")
        hits = []
        per = max(self.top_k // len(comps), 3)
        for c in comps:
            f = {}
            if c:
                f["company"] = [c]
            if yrs:
                f["year"] = yrs
            hs = r.search(self.expand(q), k=per, filters=f or None)
            if c and not yrs:
                # prefer the most recent filing of that company
                latest = max((d.get("year", "") for d in r.index.docs if d.get("company") == c), default="")
                hs = r.search(self.expand(q), k=per, filters={"company": [c]}, boost={"year": [latest]})
            hits += hs
        hits.sort(key=lambda h: -h.score)
        return hits[: max(self.top_k, per * len(comps))]

    def specialist(self, q, hits):
        matrix = defaultdict(lambda: defaultdict(list))
        for h in hits:
            comp = h.doc.get("company", "?")
            tl = " " + h.doc["text"].lower() + " "
            for cat, kws in RISK_TAXONOMY.items():
                n = sum(tl.count(k) for k in kws)
                if n:
                    matrix[comp][cat].append((n, h.cid))
        ranked = []
        for comp, cats in matrix.items():
            for cat, lst in cats.items():
                mentions = sum(n for n, _ in lst)
                ranked.append({"title": f"{comp}: {cat} risk", "type": "risk", "score": mentions,
                               "rationale": f"{mentions} risk-language mentions across {len(lst)} retrieved passages",
                               "evidence": ", ".join(sorted({c for _, c in lst}))})
        ranked.sort(key=lambda r: -r["score"])
        comps = self.companies(q)
        spec = {"recommendations": ranked[:6],
                "extras": {"risk_matrix": {c: {k: sum(n for n, _ in v) for k, v in cats.items()}
                                           for c, cats in matrix.items()},
                           "filings": sorted({h.doc["source"] for h in hits})},
                "next_actions": [
                    {"action": "Track the top-ranked risk categories as KPIs in vendor/partner reviews",
                     "why": "They are the risks the companies themselves disclose as material",
                     "kpi": "Quarterly change in disclosed risk, revenue/margin trend", "priority": "P1"},
                    {"action": "Mitigate supplier concentration: multi-vendor sourcing or contractual SLAs/price protection",
                     "why": "Supply-chain and tariff exposure flows through to customers' cost and lead time",
                     "kpi": "Lead time, price variance", "priority": "P2"},
                    {"action": "Re-check the next 10-Q/10-K for changes before major commitments",
                     "why": "Filings are point-in-time", "kpi": "Risk factors added/removed", "priority": "P3"}],
                "missing_info": [] if comps else ["Which company (Cisco, Agilent, Alphabet) and fiscal year?"]}
        return spec


# =============================================================================== Corporate responsibility
INVESTMENTS = [
    {"title": "Climate-resilient agriculture & early-warning systems",
     "query": "climate extremes resilience early warning systems agriculture adaptation",
     "kpi": "Yield stability under shocks; PoU in climate-exposed regions", "tech": True},
    {"title": "Digital agriculture & market information platforms for smallholders",
     "query": "digital technologies smallholder farmers access markets information innovation",
     "kpi": "Smallholder income; post-harvest price received", "tech": True},
    {"title": "Social protection / cash-transfer programmes",
     "query": "social protection cash transfers safety nets affordability healthy diet",
     "kpi": "Share of population unable to afford a healthy diet"},
    {"title": "Nutrition programmes for mothers & children (stunting, wasting, anaemia)",
     "query": "child stunting wasting anaemia women nutrition interventions breastfeeding",
     "kpi": "Stunting / wasting prevalence; exclusive breastfeeding rate"},
    {"title": "Food loss reduction: cold chain, storage and logistics",
     "query": "food loss supply chains storage infrastructure logistics rural urban connectivity",
     "kpi": "Food loss % along the supply chain", "tech": True},
    {"title": "Humanitarian food assistance in conflict-affected areas",
     "query": "conflict humanitarian assistance acute food insecurity crises displacement",
     "kpi": "People in acute food insecurity (IPC 3+) reached"},
    {"title": "Financing for food security and nutrition (blended finance)",
     "query": "financing to end hunger food insecurity malnutrition funding flows",
     "kpi": "Financing mobilised per beneficiary; cost-effectiveness"},
    {"title": "Peri-urban agrifood systems & healthy-diet affordability",
     "query": "urbanization agrifood systems peri-urban healthy diets cost affordability",
     "kpi": "Cost of a healthy diet vs. income in target cities"},
]


class CSRAgent(BaseAgent):
    domain, prefix, index_name = "csr", "E", "csr"
    expansions = {"war": "conflict armed conflict crises", "price": "food price inflation cost affordability",
                  "reason": "drivers causes conflict climate economic downturns", "cause": "drivers",
                  "malnutrition": "stunting wasting undernutrition", "hunger": "undernourishment prevalence",
                  "technology": "innovation digital technologies", "invest": "financing investment"}

    def filters(self, q):
        ql = q.lower()
        m = re.search(r"(?:sofi|report|edition)\s*(20\d\d)|(20\d\d)\s*(?:sofi|report|edition)", ql)
        if m:
            return {"filters": {"year": [m.group(1) or m.group(2)]}}
        yrs = re.findall(r"\b(202[34])\b", ql)
        return {"boost": {"year": yrs}} if yrs else {}

    def specialist(self, q, hits):
        ql = q.lower()
        wants_invest = any(w in ql for w in ("invest", "fund", "technology", "csr", "esg", "program", "should",
                                             "recommend", "improve", "strategy", "initiative"))
        spec = {"recommendations": [], "next_actions": [], "missing_info": [], "extras": {}}
        if wants_invest:
            r = Retriever("csr")
            generic = set(tokenize("invest investment fund funding program technology improve food security "
                                   "sustainability outcome csr esg recommend strategy initiative should"))
            qt = set(tokenize(q)) - generic
            ranked = []
            for inv in INVESTMENTS:
                ev = r.search(inv["query"] + " " + " ".join(qt), k=2, per_source=1)
                support = ev[0].score if ev else 0
                inv_t = set(tokenize(inv["title"] + " " + inv["query"]))
                rel = len(qt & inv_t) / max(len(qt), 1) if qt else 0.0
                score = 0.75 * rel + 0.25 * min(support / 40, 1)
                if inv.get("tech") and re.search(r"technolog|digital|\bai\b|data|cloud|iot", ql):
                    score += 0.15
                ranked.append({"title": inv["title"], "type": "investment", "score": round(score, 3),
                               "rationale": f"query relevance {rel:.2f}; SOFI evidence strength {support:.1f}",
                               "evidence": ev[0].citation if ev else "", "kpi": inv["kpi"],
                               "snippet": ev[0].doc["text"][:220] if ev else ""})
            ranked.sort(key=lambda x: -x["score"])
            spec["recommendations"] = ranked[:4]
            spec["next_actions"] = [
                {"action": f"Prioritise: {ranked[0]['title']}", "why": "Highest combined relevance and evidence support",
                 "kpi": ranked[0]["kpi"], "priority": "P1"},
                {"action": "Define a baseline using SOFI indicators (PoU, FIES, healthy-diet affordability) for target regions",
                 "why": "Makes impact measurable and comparable year over year", "kpi": "Baseline report", "priority": "P1"},
                {"action": "Partner with an implementing organisation and report against SDG 2 targets",
                 "why": "CSR credibility depends on measured outcomes", "kpi": "SDG 2.1 / 2.2 indicators", "priority": "P2"}]
            spec["missing_info"] = ["Target geography and beneficiary group", "Budget and time horizon",
                                    "Link to the company's core business (e.g. technology, logistics, health)"]
        return spec


# =============================================================================== Retail / market basket
class RetailAgent(BaseAgent):
    domain, prefix, index_name = "retail", "R", ""
    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            from core.basket import BasketRecommender
            cls._model = BasketRecommender()
        return cls._model

    def parse_cart(self, q: str) -> List[str]:
        m = self.model()
        ql = q.lower()
        found = []
        for item in m.items:
            name, sku = item.lower(), re.search(r"sku:\s*([^)]+)", item.lower())
            sku = sku.group(1).strip() if sku else ""
            core = re.sub(r"\(sku.*\)", "", name).strip()
            toks = [t for t in re.findall(r"[a-z0-9\-]+", core) if len(t) > 2]
            model_toks = [t for t in toks if any(c.isdigit() for c in t)]
            if (sku and sku in ql) or core in ql or (model_toks and all(t in ql for t in model_toks)):
                found.append(item)
        return found

    def run(self, q, sub_question=None):
        from core.basket import exact_cart_seen
        t0 = time.time()
        res = AgentResult(domain=self.domain, label=self.label, question=sub_question or q, mode="recommender")
        m = self.model()
        cart = self.parse_cart(q)
        recs = m.recommend(cart, k=5) if cart else []
        seen = exact_cart_seen(m.tx, cart) if cart else 0
        rules = m.top_rules(5)
        res.extras = {"cart": cart, "exact_cart_seen_in_history": seen, "top_rules": rules,
                      "recs": [r.to_dict() for r in recs]}
        res.recommendations = [{"title": r.item, "type": "cross-sell", "score": round(r.score, 3),
                                "rationale": f"confidence {r.confidence:.0%}, lift {r.lift:.2f}, "
                                             f"seen together {r.support_count}x with {', '.join(r.antecedent)}",
                                "kpi": "attach rate, average order value"} for r in recs]
        res.evidence = [{"cid": f"R{i}", "citation": f"Transaction history ({m.n} baskets): rule "
                                                      f"{{{', '.join(r.antecedent)}}} -> {r.item}",
                         "text": f"support={r.support_count}, confidence={r.confidence:.3f}, lift={r.lift:.2f}",
                         "score": r.score, "source": "market basket transactions", "page": ""}
                        for i, r in enumerate(recs, 1)]
        lines = []
        if not cart:
            lines.append("I could not identify a catalogue product in the request. Strongest associations in the "
                         "transaction history:")
            lines += [f"- If **{r['if']}** then **{r['then']}** (lift {r['lift']}, confidence {r['confidence']:.0%})"
                      for r in rules]
            res.missing_info = ["Name or SKU of the product(s) in the customer's cart"]
        elif not recs:
            lines.append(f"Cart: {', '.join(cart)}.\n\nNo reliable co-purchase pattern (support >= {m.min_count}, "
                         f"lift >= {m.min_lift}) exists for this combination, so **no recommendation is shown** rather "
                         "than an irrelevant one.")
        else:
            note = "" if seen else (" This exact combination never appeared in history; recommendations back off to "
                                    "the sub-combinations shown in *because of*.")
            lines.append(f"Cart: {', '.join(cart)}.{note}\n\n**Recommended add-ons:**")
            lines += [f"{i}. **{r.item}** - bought with {', '.join(r.antecedent)} in {r.support_count} baskets "
                      f"(confidence {r.confidence:.0%}, lift {r.lift:.2f}) [R{i}]" for i, r in enumerate(recs, 1)]
        res.answer = "\n".join(lines)
        top_lift = recs[0].lift if recs else 0
        sc = 0.3 if not recs else min(0.35 + 0.2 * (top_lift - 1) + 0.02 * recs[0].support_count, 0.8)
        res.confidence = {"score": round(sc, 2), "level": "High" if sc >= 0.7 else "Medium" if sc >= 0.48 else "Low",
                          "reasons": [f"top lift {top_lift:.2f}", "small dataset (1,000 baskets): associations are weak"]}
        res.next_actions = [{"action": "A/B test the top add-on recommendations at checkout",
                             "why": "Association rules are correlational; test causally before full rollout",
                             "kpi": "Attach rate, AOV, conversion", "priority": "P1"}]
        res.latency_s = round(time.time() - t0, 2)
        return res


AGENTS = {"technology": TechnologyAgent, "clinical": ClinicalAgent, "sec": SECAgent, "csr": CSRAgent,
          "retail": RetailAgent}
