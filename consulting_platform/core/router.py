"""Management Consulting Manager - intent detection and routing.

Three routing strategies (compared in evaluate.py):
  * lexicon   : weighted domain vocabulary / entity matching
  * prototype : TF-IDF nearest-centroid classifier trained on seed requests (a tiny supervised model)
  * llm       : zero-shot LLM classification (when an LLM is configured)
The default 'hybrid' strategy fuses lexicon + prototype, and lets the LLM decide when available.
Multi-label output enables cross-domain orchestration.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from core.text import tokenize

DOMAINS = list(config.DOMAINS.keys())

# (phrase, weight) -- phrases are tokenised the same way as queries
LEXICON: Dict[str, Dict[str, float]] = {
    "technology": {
        "cisco": 1.0, "router": 2.0, "routers": 2.0, "switch": 1.2, "catalyst": 1.5, "isr": 2.0, "asr": 2.0,
        "isr4331": 2.5, "isr4451": 2.5, "isr4321": 2.5, "isr4351": 2.5, "isr4431": 2.5, "isr4461": 2.5,
        "8300": 1.5, "8200": 1.5, "8000v": 2.0, "asr 1000": 2.0, "sd-wan": 2.0, "wan": 1.2, "ipsec": 1.5, "vpn": 1.2,
        "firmware": 1.5, "ios xe": 2.0, "outage": 1.2, "connectivity": 1.5, "network": 1.2, "interface": 1.0,
        "packet loss": 2.0, "latency": 1.0, "throughput": 1.0, "bandwidth": 1.0, "google cloud": 1.8, "gcp": 2.0,
        "model context protocol": 2.0, "mcp": 1.8, "retrieval-augmented generation": 1.5, "rag": 1.2,
        "ibm": 1.5, "data center": 1.2, "failover": 1.5, "redundancy": 1.0, "power supply": 1.5, "fan tray": 1.5,
        "module": 0.8, "transceiver": 1.8, "sfp": 1.8, "replacement part": 1.5, "replacement component": 1.2,
        "spare": 1.0, "edge platform": 1.5, "cloud": 0.8, "it infrastructure": 1.5, "bgp": 2.0, "ospf": 2.0,
        "reboot": 1.2, "crash": 1.0, "cpu": 0.8, "memory leak": 1.5, "upgrade": 0.6, "ai agent": 0.8,
        "llm": 0.6, "enterprise software": 1.0, "server": 0.8, "intermittent": 0.8, "link flap": 2.0,
    },
    "clinical": {
        "agilent instrument": 2.0, "clinical": 2.0, "patient": 1.8, "diagnostic": 1.8, "diagnostics": 1.8,
        "laboratory": 2.0, "lab": 1.5, "pathology": 2.0, "slide scanner": 2.5, "nanozoomer": 2.5, "seahorse": 2.5,
        "cell metabolic": 2.5, "cell analysis": 2.0, "stain": 2.0, "stains": 2.0, "diabetes": 2.0, "glucose": 2.0,
        "covid": 2.0, "covid-19": 2.0, "heart attack": 2.0, "heart disease": 2.0, "cardiac": 1.8, "knee": 2.0,
        "exoskeleton": 2.0, "arthroplasty": 2.0, "clinical trial": 2.5, "trial": 1.2, "trials": 1.2,
        "disease": 1.2, "medical": 1.8, "hospital": 1.8, "biomarker": 2.0, "assay": 2.0, "imaging": 1.0,
        "x-ray": 1.5, "ct scan": 1.5, "healthcare": 1.8, "health": 1.0, "wearable": 1.0, "glycemic": 2.0,
        "insulin": 2.0, "surgery": 1.8, "rehabilitation": 1.8, "instrument": 1.0, "reagent": 1.8,
        "histology": 2.0, "specimen": 1.5, "tissue": 1.5, "sample throughput": 1.5, "ivd": 2.0,
    },
    "sec": {
        "10-k": 2.5, "sec": 2.0, "filing": 2.0, "filings": 2.0, "annual report": 2.0, "revenue": 1.8,
        "earnings": 1.8, "financial": 1.8, "financials": 2.0, "risk factors": 2.5, "risks": 0.8, "competitive": 1.2,
        "competition": 1.2, "competitor": 1.2, "gross margin": 2.0, "margin": 1.2, "net income": 2.0,
        "cash flow": 2.0, "debt": 1.5, "dividend": 2.0, "repurchase": 2.0, "guidance": 1.0, "fiscal": 1.5,
        "segment": 1.2, "acquisition": 1.2, "investor": 1.5, "valuation": 1.5, "tariff": 1.5, "tariffs": 1.5,
        "alphabet": 1.8, "stock": 1.2, "liquidity": 1.8, "profit": 1.5, "profitability": 1.5,
        "operating income": 2.0, "backlog": 1.5, "remaining performance obligations": 2.0, "eps": 2.0,
        "shareholder": 1.5, "lawsuit": 1.2, "legal proceedings": 2.0, "cybersecurity risk": 1.0,
        "operational risk": 1.5, "financial risk": 2.0, "headcount": 1.0, "employees": 0.8, "r&d": 1.2,
        "research and development spending": 1.5, "invest in": 0.5, "cisco": 0.5, "agilent": 0.5, "google": 0.3,
    },
    "csr": {
        "food security": 3.0, "food insecurity": 3.0, "hunger": 2.5, "malnutrition": 2.5, "nutrition": 2.0,
        "undernourishment": 2.5, "undernourished": 2.5, "sofi": 2.5, "fao": 2.5, "sustainability": 2.0,
        "sustainable": 1.5, "esg": 2.5, "csr": 2.5, "corporate social responsibility": 3.0,
        "corporate responsibility": 2.5, "climate": 1.5, "agriculture": 2.0, "agrifood": 2.5, "famine": 2.5,
        "poverty": 2.0, "stunting": 2.5, "wasting": 2.5, "obesity": 1.5, "food prices": 2.5, "food price": 2.5,
        "healthy diet": 2.5, "war zones": 2.0, "war": 1.2, "conflict": 1.0, "rural": 1.5, "smallholder": 2.0,
        "urbanization": 2.0, "social impact": 2.0, "philanthropy": 2.0, "food": 1.0, "crop": 1.5,
        "farmers": 1.8, "humanitarian": 1.8, "drought": 1.5, "sdg": 2.0, "inflation": 0.8, "anaemia": 2.0,
    },
    "retail": {
        "cart": 2.5, "basket": 2.5, "market basket": 3.0, "checkout": 2.5, "bought together": 3.0,
        "frequently purchased": 3.0, "purchased together": 3.0, "cross-sell": 2.5, "upsell": 2.0, "bundle": 1.5,
        "sku": 1.5, "nvr": 2.0, "network camera": 2.0, "ip camera": 2.0, "detector": 1.5, "alarm": 1.5,
        "smoke detector": 2.0, "carbon monoxide": 2.0, "motion detector": 2.0, "bosch": 2.0, "axis": 1.5,
        "dahua": 2.0, "hikvision": 2.0, "honeywell": 1.5, "dsc": 1.5, "hanwha": 2.0, "pelco": 2.0,
        "interlogix": 2.0, "glassbreak": 2.0, "control panel": 1.5, "wireless key": 2.0, "security equipment": 2.0,
        "retailer": 1.8, "shopper": 2.0, "customers also buy": 3.0, "add to cart": 3.0,
    },
}

SEEDS: Dict[str, List[str]] = {
    "technology": [
        "Our Cisco router keeps dropping connections, which parts should we replace?",
        "Compare Catalyst 8300 and ISR 4000 for branch SD-WAN throughput",
        "Which Cisco edge platform supports IPsec throughput above 1 Gbps?",
        "What is the Model Context Protocol on Google Cloud and how do agents use it?",
        "How does retrieval augmented generation work on Google Cloud?",
        "Recommend a virtual router for public cloud deployment",
        "Our WAN link flaps and the router reboots randomly, what diagnostics do we need?",
        "What memory and flash does the ASR 1000 route processor support?",
        "Should we migrate our branch routers to Catalyst 8200?",
        "Network outage prevention and redundancy best practices for enterprise routers",
        "Which modules and interfaces fit the Cisco 4331 router?",
        "Design an IT infrastructure for an AI consulting platform on cloud",
    ],
    "clinical": [
        "Which Agilent instrument best fits our pathology lab digital slide scanning needs?",
        "Recommend a cell metabolic analysis platform for our research lab",
        "What machine learning methods predict heart attacks from wearable data?",
        "Summarize clinical trials on continuous glucose monitoring for type 2 diabetes",
        "How is deep learning used for COVID-19 detection from lung images?",
        "What are options for knee rehabilitation exoskeletons?",
        "Our laboratory needs special stains automation, what do you recommend?",
        "Find recruiting clinical studies for knee injuries",
        "Which biomarkers relate to type 1 and type 2 diabetes?",
        "IoT monitoring for post-operative knee fracture patients",
        "Diagnostic instrument replacement component for our clinical lab",
        "Explain AI for improving glycemic control in diabetes patients",
    ],
    "sec": [
        "What are the main risk factors in Cisco's 10-K?",
        "Compare Agilent and Cisco revenue and gross margin for fiscal 2025",
        "What competitive risks does Alphabet disclose in its annual report?",
        "How did tariffs affect Agilent's cost of revenue?",
        "Summarize Cisco's cash flow, dividends and share repurchases",
        "What are the operational risks associated with these companies?",
        "What acquisitions did Cisco complete according to its filing?",
        "Alphabet net income and operating margin trend",
        "What legal proceedings are disclosed in the SEC filing?",
        "Agilent segment performance life sciences diagnostics",
        "What is Cisco's remaining performance obligation and backlog?",
        "Financial health and liquidity of Agilent",
    ],
    "csr": [
        "List major food insecurity reasons in 2023",
        "Explain malnutrition in war zones",
        "Explain the impact of increased food prices on food security",
        "How many people faced hunger in 2023 according to SOFI 2024?",
        "How does urbanization affect agrifood systems and diets?",
        "Which technology investments could improve food security and sustainability outcomes?",
        "What financing is needed to end hunger and malnutrition?",
        "Trends in child stunting and wasting worldwide",
        "Cost and affordability of a healthy diet",
        "Our CSR program wants to support smallholder farmers, what should we fund?",
        "Climate extremes and conflict as drivers of hunger",
        "ESG strategy for a company focused on sustainable agriculture",
    ],
    "retail": [
        "Customer added a Bosch D7050 detector to the cart, what else should we recommend?",
        "Which items are frequently bought together with the Dahua NVR?",
        "Recommend cross-sell products at checkout for a Hikvision camera",
        "What should we bundle with the DSC PowerSeries Neo alarm kit?",
        "Market basket analysis for our security equipment store",
        "Shopper has an Axis network camera and a Bosch control panel in the cart",
        "Which smoke detector do customers also buy with a carbon monoxide detector?",
        "Top product associations in our retail transaction history",
        "Build a bundle offer for an IP camera and NVR",
        "What add-on items increase basket size for alarm panels?",
        "Recommend items for a cart with Honeywell and Hanwha products",
        "Checkout recommendations for security equipment",
    ],
}

INTENTS = {
    "troubleshooting": ["fail", "failure", "outage", "drop", "intermittent", "error", "crash", "reboot", "not working",
                        "issue", "problem", "fault", "flap", "broken", "degrad"],
    "product_recommendation": ["which", "recommend", "replacement", "order", "buy", "choose", "best", "select",
                               "fit", "suitable", "component", "instrument", "part"],
    "risk_analysis": ["risk", "exposure", "threat", "vulnerab", "uncertain"],
    "financial_analysis": ["revenue", "margin", "income", "cash", "financial", "earnings", "profit", "dividend"],
    "investment_strategy": ["invest", "fund", "strategy", "prioritize", "roadmap", "should we", "plan"],
    "research_summary": ["explain", "summar", "list", "what are", "describe", "overview", "trend", "reason"],
    "cross_sell": ["cart", "basket", "bought", "bundle", "checkout", "also buy"],
}


@dataclass
class RouteDecision:
    domains: List[str]
    primary: str
    probs: Dict[str, float]
    intent: str
    strategy: str
    reason: str = ""
    cross_domain: bool = False
    signals: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__.copy()


class Router:
    def __init__(self, llm=None):
        self.llm = llm
        self._lex = {d: [(tokenize(p), w) for p, w in terms.items()] for d, terms in LEXICON.items()}
        # prototype classifier (TF-IDF centroids)
        docs = [(d, tokenize(s)) for d, ss in SEEDS.items() for s in ss]
        df = Counter(t for _, toks in docs for t in set(toks))
        n = len(docs)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self._centroids = {}
        for d in DOMAINS:
            vec = Counter()
            for dd, toks in docs:
                if dd == d:
                    for t, v in self._tfidf(toks).items():
                        vec[t] += v
            self._centroids[d] = self._norm(vec)

    # ------------------------------------------------------------- helpers
    def _tfidf(self, toks):
        tf = Counter(toks)
        return self._norm({t: (1 + math.log(c)) * self._idf.get(t, math.log(len(SEEDS) * 12) + 1)
                           for t, c in tf.items()})

    @staticmethod
    def _norm(vec):
        s = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / s for t, v in vec.items()}

    def lexicon_scores(self, query: str):
        toks = tokenize(query)
        joined = " " + " ".join(toks) + " "
        scores, signals = {}, {}
        for d, terms in self._lex.items():
            s, hit = 0.0, []
            for ptoks, w in terms:
                if not ptoks:
                    continue
                if f" {' '.join(ptoks)} " in joined:
                    s += w
                    hit.append(" ".join(ptoks))
            scores[d], signals[d] = s, hit
        return scores, signals

    def prototype_scores(self, query: str):
        q = self._tfidf(tokenize(query))
        return {d: sum(v * c.get(t, 0.0) for t, v in q.items()) for d, c in self._centroids.items()}

    @staticmethod
    def detect_intent(query: str) -> str:
        q = query.lower()
        best, score = "research_summary", 0
        for intent, kws in INTENTS.items():
            s = sum(1 for k in kws if k in q)
            if s > score:
                best, score = intent, s
        return best

    # ------------------------------------------------------------- routing
    def route_heuristic(self, query: str, strategy: str = "hybrid") -> RouteDecision:
        lex, signals = self.lexicon_scores(query)
        proto = self.prototype_scores(query)
        mx_l = max(lex.values()) or 1.0
        mx_p = max(proto.values()) or 1.0
        if strategy == "lexicon":
            comb = {d: lex[d] / mx_l for d in DOMAINS}
        elif strategy == "prototype":
            comb = {d: proto[d] / mx_p for d in DOMAINS}
        else:
            comb = {d: 0.6 * lex[d] / mx_l + 0.4 * proto[d] / mx_p for d in DOMAINS}
        z = {d: math.exp(4.0 * v) for d, v in comb.items()}
        tot = sum(z.values())
        probs = {d: round(v / tot, 3) for d, v in z.items()}
        ranked = sorted(DOMAINS, key=lambda d: -probs[d])
        primary = ranked[0]
        # multi-label: secondary domains need both probability mass and real lexical evidence
        chosen = [primary] + [d for d in ranked[1:]
                              if (probs[d] >= 0.22 and lex[d] >= 1.5 and probs[d] >= 0.45 * probs[primary])
                              or (lex[d] >= 2.5 and lex[d] >= 0.35 * lex[primary])]
        chosen = chosen[:3]
        reason = "; ".join(f"{d}: {', '.join(signals[d][:4])}" for d in chosen if signals[d]) or \
            "semantic similarity to seed requests"
        return RouteDecision(domains=chosen, primary=primary, probs=probs, intent=self.detect_intent(query),
                             strategy=strategy, reason=reason, cross_domain=len(chosen) > 1,
                             signals={d: s for d, s in signals.items() if s})

    def route_llm(self, query: str, heuristic: Optional[RouteDecision] = None) -> RouteDecision:
        desc = "\n".join(f"- {d}: {m['label']} -- knowledge: {m['sources']}" for d, m in config.DOMAINS.items())
        prompt = f"""You are the Management Consulting Manager Agent. Route the customer request to the specialist
agents that are REQUIRED to answer it. Choose one domain unless the request clearly spans several
(e.g. technical + financial + sustainability). Available agents:
{desc}

Customer request: \"\"\"{query}\"\"\"

Return JSON: {{"domains": [ordered list, most important first], "intent": one of
{list(INTENTS.keys())}, "reason": "<one sentence>"}}"""
        out = self.llm.generate_json(prompt)
        doms = [d for d in out.get("domains", []) if d in DOMAINS][:3]
        if not doms:
            raise ValueError("LLM returned no valid domain")
        probs = heuristic.probs if heuristic else {d: (1.0 if d in doms else 0.0) for d in DOMAINS}
        return RouteDecision(domains=doms, primary=doms[0], probs=probs,
                             intent=out.get("intent") if out.get("intent") in INTENTS else self.detect_intent(query),
                             strategy="llm", reason=out.get("reason", ""), cross_domain=len(doms) > 1,
                             signals=heuristic.signals if heuristic else {})

    def route(self, query: str, strategy: str = "auto", force: Optional[List[str]] = None) -> RouteDecision:
        h = self.route_heuristic(query, "hybrid" if strategy in ("auto", "llm") else strategy)
        if force:
            h.domains, h.primary, h.cross_domain, h.strategy = force, force[0], len(force) > 1, "manual"
            return h
        if strategy in ("auto", "llm") and self.llm is not None and self.llm.available:
            try:
                return self.route_llm(query, h)
            except Exception as e:
                h.reason += f" (LLM routing unavailable: {str(e)[:80]})"
        return h
