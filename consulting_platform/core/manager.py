"""Management Consulting Manager Agent: route -> (decompose) -> run specialists in parallel -> consolidate."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from core.agents import AGENTS, AgentResult
from core.llm import LLM, LLMError, get_llm
from core.router import RouteDecision, Router

CONSOLIDATE_SYSTEM = """You are the Management Consulting Manager Agent. You turn specialist agent findings into a
client-ready consulting recommendation. Use ONLY the specialist findings; keep their citation tags exactly
(e.g. [T2], [F1], [E3], [C1], [R1]). Distinguish confirmed facts from hypotheses. Be concrete and measurable."""

CONSOLIDATE_FORMAT = """Write Markdown with exactly these sections:
## Executive summary
(3-4 sentences: the situation and the single most important recommendation)
## Findings
(bullets grouped by specialist; mark **Confirmed** vs **Hypothesis** where relevant; keep citations)
## Recommended actions
| Priority | Action | Why (evidence) | How to measure (KPI) | Timeline |
(4-7 rows, P1 first)
## Risks & dependencies
(bullets)
## Information needed from the client
(bullets)
## Confidence
(one line: High/Medium/Low and why)"""


@dataclass
class ConsultingResponse:
    query: str
    route: RouteDecision
    agents: List[AgentResult]
    recommendation: str
    confidence: Dict
    llm: str
    latency_s: float
    sub_questions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {"query": self.query, "route": self.route.to_dict(), "agents": [a.to_dict() for a in self.agents],
                "recommendation": self.recommendation, "confidence": self.confidence, "llm": self.llm,
                "latency_s": self.latency_s, "sub_questions": self.sub_questions}


class ConsultingManager:
    def __init__(self, llm: Optional[LLM] = None, top_k: int = config.TOP_K):
        self.llm = llm if llm is not None else get_llm()
        self.router = Router(self.llm)
        self.top_k = top_k

    # ------------------------------------------------------------------ decomposition
    def decompose(self, query: str, domains: List[str]) -> Dict[str, str]:
        if len(domains) < 2 or not self.llm.available:
            return {d: query for d in domains}
        desc = "\n".join(f"- {d}: {config.DOMAINS[d]['label']} ({config.DOMAINS[d]['sources']})" for d in domains)
        prompt = (f"Split this client request into one focused sub-question per specialist so that together they "
                  f"answer it.\nSpecialists:\n{desc}\n\nRequest: \"{query}\"\n\nReturn JSON "
                  f"{{\"<domain>\": \"<sub-question>\"}} using exactly these keys: {domains}")
        try:
            out = self.llm.generate_json(prompt)
            return {d: (out.get(d) or query) for d in domains}
        except Exception:
            return {d: query for d in domains}

    # ------------------------------------------------------------------ main entry
    def ask(self, query: str, strategy: str = "auto", force_domains: Optional[List[str]] = None,
            consolidate: bool = True) -> ConsultingResponse:
        t0 = time.time()
        route = self.router.route(query, strategy=strategy, force=force_domains)
        subs = self.decompose(query, route.domains)

        def run(d):
            agent = AGENTS[d](self.llm, top_k=self.top_k)
            return agent.run(subs.get(d, query))

        with ThreadPoolExecutor(max_workers=len(route.domains)) as ex:
            results = list(ex.map(run, route.domains))

        conf = self.overall_confidence(route, results)
        rec = self.consolidate(query, route, results, conf) if consolidate else ""
        return ConsultingResponse(query=query, route=route, agents=results, recommendation=rec, confidence=conf,
                                  llm=self.llm.label, latency_s=round(time.time() - t0, 2), sub_questions=subs)

    # ------------------------------------------------------------------ confidence
    @staticmethod
    def overall_confidence(route: RouteDecision, results: List[AgentResult]) -> Dict:
        if not results:
            return {"score": 0, "level": "Low", "reasons": []}
        agent_scores = [r.confidence.get("score", 0) for r in results]
        route_conf = route.probs.get(route.primary, 0.5) if route.strategy != "llm" else 0.9
        score = 0.8 * (sum(agent_scores) / len(agent_scores)) + 0.2 * min(route_conf * 1.5, 1)
        score = min(score, min(agent_scores) + 0.25)            # one weak agent limits the whole answer
        level = "High" if score >= 0.7 else "Medium" if score >= 0.48 else "Low"
        return {"score": round(score, 2), "level": level,
                "reasons": [f"routing confidence {route_conf:.2f} ({route.strategy})"] +
                           [f"{r.label}: {r.confidence.get('level')} ({r.confidence.get('score')})" for r in results]}

    # ------------------------------------------------------------------ consolidation
    def consolidate(self, query: str, route: RouteDecision, results: List[AgentResult], conf: Dict) -> str:
        if self.llm.available:
            blocks = []
            for r in results:
                acts = "\n".join(f"- [{a.get('priority')}] {a.get('action')} | why: {a.get('why')} | KPI: {a.get('kpi')}"
                                 for a in r.next_actions)
                blocks.append(f"=== {r.label} (confidence {r.confidence.get('level')}) ===\n{r.answer}\n"
                              f"Suggested next actions:\n{acts}\nMissing info: {'; '.join(r.missing_info)}")
            prompt = (f"Client request: {query}\nRouting: {route.domains} (intent: {route.intent})\n"
                      f"Platform confidence: {conf['level']} ({conf['score']})\n\n" + "\n\n".join(blocks) +
                      f"\n\n{CONSOLIDATE_FORMAT}")
            try:
                return self.llm.generate(prompt, system=CONSOLIDATE_SYSTEM, max_tokens=2200)
            except LLMError:
                pass
        return self.template_consolidation(query, route, results, conf)

    @staticmethod
    def template_consolidation(query, route, results, conf) -> str:
        names = ", ".join(r.label for r in results)
        out = ["## Executive summary",
               f"Request routed to **{names}** (intent: *{route.intent.replace('_', ' ')}*). "
               f"Overall confidence: **{conf['level']}** ({conf['score']}). "
               "The recommended actions below are generated from the specialists' evidence and recommenders.",
               "", "## Findings"]
        for r in results:
            out.append(f"**{r.label}** - confidence {r.confidence.get('level')}")
            hyps = r.extras.get("hypotheses") if r.extras else None
            if hyps:
                for h in hyps[:3]:
                    out.append(f"- *{h['status']}*: {h['hypothesis']} (likelihood {h['likelihood']})")
            for rec in r.recommendations[:3]:
                ev = f" [{rec['evidence'][:80]}]" if rec.get("evidence") else ""
                out.append(f"- Recommended: **{rec['title']}** - {rec.get('rationale','')[:140]}{ev}")
            for e in r.evidence[:2]:
                out.append(f"- Evidence [{e.get('cid')}]: {' '.join(str(e.get('text', '')).split())[:200]}...")
        out += ["", "## Recommended actions", "| Priority | Action | Why | How to measure (KPI) | Owner |",
                "|---|---|---|---|---|"]
        rows = sorted([(str(a.get("priority", "P3")), a, r.label) for r in results for a in r.next_actions],
                      key=lambda x: x[0])
        for p, a, lab in rows[:8]:
            out.append(f"| {p} | {a.get('action', '')} | {a.get('why', '')} | {a.get('kpi', '')} | {lab} |")
        miss = [m for r in results for m in r.missing_info]
        if miss:
            out += ["", "## Information needed from the client"] + [f"- {m}" for m in dict.fromkeys(miss)]
        out += ["", "## Confidence", f"{conf['level']} - " + "; ".join(conf["reasons"])]
        return "\n".join(out)
