"""Agentic layer: ReAct specialist agents, a planning Manager, a Critic, and conversation memory.

Manager:  plan (which agents, which tasks, dependencies)  ->  delegate (parallel waves, dependent tasks receive
          upstream findings)  ->  review (are success criteria met? issue follow-up tasks)  ->  consolidate
          ->  Critic review  ->  revise if needed  ->  remember the turn.
Agent:    loop { thought -> tool call -> observation }  ->  final_answer  ->  self-check (citations, grounding,
          KPIs, hypothesis labelling)  ->  revise / gather more evidence if the check fails.
Without an LLM the same interface runs the deterministic pipeline and still produces a trace.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from core.agents import AGENTS, AgentResult
from core.llm import LLM, LLMError, get_llm
from core.manager import ConsultingManager, ConsultingResponse
from core.quality import response_metrics, self_check
from core.router import RouteDecision
from core.tools import AgentContext, get_tools
from core.trace import Trace

PREFIX = {"technology": "T", "clinical": "C", "sec": "F", "csr": "E", "retail": "R"}
SELF_CONF = {"high": 0.85, "medium": 0.6, "low": 0.35}


def _parse_json(text: str) -> Dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object")
    return json.loads(m.group(0))


def _short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."


# =========================================================================================== ReAct agent
AGENT_SYSTEM = """You are the {label} of an AI management-consulting firm. Knowledge base: {sources}.
You solve the task by reasoning step by step and calling tools. Reply with exactly ONE JSON object per turn:
{{"thought": "<what you know, what is missing, what you will do next>", "action": "<tool name or final_answer>", "args": {{...}}}}

Tools:
{tools}
- final_answer(answer, next_actions, missing_info, confidence)
    answer: Markdown with sections "### Answer", "### Key evidence", "### Recommendations", "### Gaps & assumptions".
            Cite supporting passages inline, e.g. [{p}1][{p}3].
    next_actions: list of {{"action": "...", "why": "...", "kpi": "how to measure", "priority": "P1|P2|P3"}}
    missing_info: list of strings (what the client must provide)
    confidence: "High" | "Medium" | "Low"

Rules:
- Gather evidence with tools before final_answer. If results are weak or off-topic, refine keywords or filters and search again.
- Use the specialised recommender tools when the task asks what to choose, buy, fund or fix.
- Cite only passage ids that appear in observations. Never invent figures, part numbers, products or URLs.
- Clearly separate CONFIRMED facts from HYPOTHESES. If evidence is insufficient, say so and list what is needed.
{extra}- Budget: at most {max_steps} tool calls, then final_answer."""


class ReActAgent:
    def __init__(self, domain: str, llm: LLM, trace: Trace, max_steps: int = 5, max_revisions: int = 1,
                 top_k: int = config.TOP_K):
        self.domain, self.llm, self.trace = domain, llm, trace
        self.max_steps, self.max_revisions, self.top_k = max_steps, max_revisions, top_k
        self.base = AGENTS[domain](None, top_k=top_k)          # reuses query expansion / confidence logic
        self.label = self.base.label
        self.tools = get_tools(domain)

    def system(self) -> str:
        return AGENT_SYSTEM.format(label=self.label, sources=self.base.sources, p=PREFIX[self.domain],
                                   tools="\n".join(t.spec() for t in self.tools.values()),
                                   extra=(self.base.extra_rules + "\n") if self.base.extra_rules else "",
                                   max_steps=self.max_steps)

    def _prompt(self, task, client_request, upstream, memory, scratch):
        parts = [f"TASK FROM MANAGER: {task}", f"ORIGINAL CLIENT REQUEST: {client_request}"]
        if memory:
            parts.append(f"CONVERSATION SO FAR:\n{memory}")
        if upstream:
            parts.append(f"FINDINGS FROM OTHER AGENTS (you may build on them):\n{upstream}")
        sp = []
        for i, s in enumerate(scratch):
            full = i >= len(scratch) - 2                      # keep the last two observations in full
            obs = s.get("observation", "")
            sp.append(f"Step {i + 1}\nThought: {s.get('thought', '')}\nAction: {s.get('action')}"
                      f"({json.dumps(s.get('args', {}))[:300]})\nObservation: {obs if full else _short(obs, 600)}")
        parts.append("SCRATCHPAD:\n" + ("\n\n".join(sp) if sp else "(empty - start by gathering evidence)"))
        used = sum(1 for s in scratch if not str(s.get("action", "")).startswith("final_answer"))
        parts.append(f"Tool calls used: {used}/{self.max_steps}. Respond with the next JSON object.")
        return "\n\n".join(parts)

    def _decide(self, prompt: str) -> Dict:
        txt = self.llm.generate(prompt, system=self.system(), temperature=0.1, max_tokens=2200, json_mode=True)
        try:
            return _parse_json(txt)
        except Exception:
            txt = self.llm.generate(prompt + "\n\nYour previous reply was not valid JSON. Reply with ONE JSON object only.",
                                    system=self.system(), temperature=0.0, max_tokens=2200, json_mode=True)
            return _parse_json(txt)

    def run(self, task: str, client_request: str, upstream: str = "", memory: str = "",
            allowed_ids: Optional[Dict[str, str]] = None) -> AgentResult:
        t0 = time.time()
        actor = self.label
        ctx = AgentContext(domain=self.domain, prefix=PREFIX[self.domain])
        res = AgentResult(domain=self.domain, label=self.label, question=client_request, task=task, mode="agentic")
        scratch: List[Dict] = []
        final = None
        revisions = 0
        budget = self.max_steps + 3                               # tool calls + final answer + revisions
        try:
            for _ in range(budget):
                used = sum(1 for s in scratch if not str(s.get("action", "")).startswith("final_answer"))
                prompt = self._prompt(task, client_request, upstream, memory, scratch)
                if used >= self.max_steps:
                    prompt += "\nTool budget exhausted: you MUST return final_answer now."
                d = self._decide(prompt)
                thought, action, args = d.get("thought", ""), str(d.get("action", "")).strip(), d.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                if thought:
                    self.trace.log(actor, "thought", _short(thought, 220))
                if action == "final_answer":
                    final = args
                    ans = str(final.get("answer", ""))
                    evid = [{"cid": h.cid, "text": h.doc["text"]} for h in ctx.hits]
                    if allowed_ids:
                        evid += [{"cid": c, "text": t} for c, t in allowed_ids.items()]
                    chk = self_check(ans, evid, task, final.get("next_actions"),
                                     hypotheses=bool(ctx.extras.get("hypotheses")))
                    res.self_check = chk
                    if chk["passed"] or revisions >= self.max_revisions:
                        self.trace.log(actor, "reflection",
                                       "Self-check passed" if chk["passed"] else
                                       f"Self-check still has issues after revision: {'; '.join(chk['issues'])}",
                                       chk["metrics"])
                        break
                    revisions += 1
                    self.trace.log(actor, "reflection", "Self-check failed -> revising", chk["issues"])
                    scratch.append({"thought": thought, "action": "final_answer (draft)", "args": {},
                                    "observation": "SELF-CHECK FAILED:\n- " + "\n- ".join(chk["issues"]) +
                                                   "\nFix these issues: gather more evidence if needed, then return a "
                                                   "corrected final_answer."})
                    continue
                tool = self.tools.get(action)
                if tool is None:
                    obs = f"Unknown tool '{action}'. Available: {', '.join(self.tools)} or final_answer."
                    res.tool_errors += 1
                    self.trace.log(actor, "error", f"Unknown tool '{action}' - agent told to choose a valid tool")
                else:
                    self.trace.log(actor, "tool", f"{action}({_short(json.dumps(args), 160)})")
                    try:
                        obs = tool(ctx, args)
                    except Exception as e:
                        obs = f"Tool error: {type(e).__name__}: {e}"
                        res.tool_errors += 1
                    res.tool_calls += 1
                    self.trace.log(actor, "observation", _short(obs, 260))
                scratch.append({"thought": thought, "action": action, "args": args, "observation": obs[:3500]})
            if final is None:
                raise LLMError("agent did not produce a final answer within its budget")
        except (LLMError, ValueError, json.JSONDecodeError) as e:
            self.trace.log(actor, "fallback", f"Switching to deterministic pipeline: {_short(e, 120)}")
            fb = self.base.run(task)
            fb.steps, fb.task, fb.mode = scratch, task, "fallback (pipeline)"
            fb.tool_calls, fb.tool_errors = res.tool_calls, res.tool_errors
            fb.latency_s = round(time.time() - t0, 2)
            return fb

        res.answer = str(final.get("answer", ""))
        res.next_actions = [a for a in (final.get("next_actions") or []) if isinstance(a, dict)]
        res.missing_info = list(dict.fromkeys([str(m) for m in (final.get("missing_info") or [])] + ctx.missing_info))
        res.evidence = [h.to_dict() for h in ctx.hits]
        res.recommendations = ctx.recommendations
        res.extras = ctx.extras
        res.steps = scratch
        res.revisions = revisions
        res.confidence = self.confidence(task, ctx, final, res.self_check)
        res.latency_s = round(time.time() - t0, 2)
        self.trace.log(actor, "final", f"Answer ready ({res.tool_calls} tool calls, {revisions} revision(s), "
                                       f"confidence {res.confidence['level']})")
        return res

    def confidence(self, task, ctx, final, chk) -> Dict:
        hits = sorted(ctx.hits, key=lambda h: -h.score)
        spec = {}
        hyps = ctx.extras.get("hypotheses")
        if hyps and not any("SUPPORTED" in h["status"] for h in hyps):
            spec = {"confidence_cap": 0.55, "cap_reason": "root cause not confirmed by diagnostics"}
        det = self.base.assess(task, hits, spec) if hits else {"score": 0.1, "reasons": ["no evidence"]}
        selfc = SELF_CONF.get(str(final.get("confidence", "medium")).lower(), 0.6)
        score = 0.65 * det["score"] + 0.35 * selfc
        if chk and not chk.get("passed", True):
            score -= 0.1
        if spec:
            score = min(score, spec["confidence_cap"])
        level = "High" if score >= 0.7 else "Medium" if score >= 0.48 else "Low"
        return {"score": round(score, 2), "level": level,
                "reasons": det.get("reasons", []) + [f"agent self-assessment: {final.get('confidence')}",
                                                     "self-check passed" if chk.get("passed") else "self-check issues remain"]}


# =========================================================================================== Critic
CRITIC_SYSTEM = """You are the independent Quality Reviewer (engagement partner) of a consulting firm. You check a
draft recommendation before it goes to the client. Be strict but fair. Reply with ONE JSON object."""

CRITIC_PROMPT = """CLIENT REQUEST: {q}

DRAFT RECOMMENDATION:
{rec}

EVIDENCE AVAILABLE (id: excerpt):
{ev}

AUTOMATED CHECKS: {checks}

Review for: (1) does it answer the request, (2) every factual claim supported by cited evidence, (3) hypotheses not
presented as facts, (4) actions are specific with KPIs and priorities, (5) risks and missing information addressed.
Return JSON: {{"verdict": "approve" | "revise", "scores": {{"relevance": 1-5, "groundedness": 1-5,
"actionability": 1-5, "risk_awareness": 1-5}}, "issues": ["..."], "required_changes": ["..."]}}"""


class Critic:
    def __init__(self, llm: LLM, trace: Trace):
        self.llm, self.trace = llm, trace

    def review(self, query: str, recommendation: str, results: List[AgentResult]) -> Dict:
        ev = [e for r in results for e in r.evidence]
        hyps = any((r.extras or {}).get("hypotheses") for r in results)
        chk = self_check(recommendation, ev, query, None, hypotheses=hyps)
        out = {"verdict": "approve" if chk["passed"] else ("revise" if self.llm.available else "flagged"),
               "issues": chk["issues"],
               "metrics": chk["metrics"], "scores": {}, "required_changes": [], "reviewer": "rules"}
        if self.llm.available:
            ev_txt = "\n".join(f"[{e['cid']}] {_short(e.get('text', ''), 220)}" for e in ev[:30])
            try:
                j = _parse_json(self.llm.generate(
                    CRITIC_PROMPT.format(q=query, rec=recommendation[:7000], ev=ev_txt,
                                         checks=json.dumps(chk["issues"]) or "none"),
                    system=CRITIC_SYSTEM, temperature=0.0, max_tokens=900, json_mode=True))
                out["scores"] = j.get("scores", {})
                out["issues"] = list(dict.fromkeys(chk["issues"] + list(j.get("issues", []))))
                out["required_changes"] = j.get("required_changes", [])
                out["verdict"] = "revise" if (j.get("verdict") == "revise" or not chk["passed"]) else "approve"
                out["reviewer"] = "llm + rules"
            except Exception as e:
                out["issues"].append(f"(LLM critic unavailable: {_short(e, 60)})")
        self.trace.log("Critic", "critique", f"Verdict: {out['verdict'].upper()}"
                       + (f" - {len(out['issues'])} issue(s)" if out["issues"] else ""),
                       {"issues": out["issues"], "scores": out["scores"]})
        return out


# =========================================================================================== Memory
@dataclass
class ConversationMemory:
    turns: List[Dict] = field(default_factory=list)
    max_turns: int = 4

    def add(self, query: str, resp: "AgenticResponse"):
        summ = resp.recommendation
        m = re.search(r"## Executive summary\s*(.+?)(?:\n## |\Z)", summ, re.S)
        self.turns.append({"query": query, "agents": [a.domain for a in resp.agents],
                           "summary": _short(m.group(1) if m else summ, 500)})
        self.turns = self.turns[-self.max_turns:]

    def context(self) -> str:
        return "\n".join(f"Turn {i + 1}: client asked \"{t['query']}\" -> agents {t['agents']}; outcome: {t['summary']}"
                         for i, t in enumerate(self.turns))

    def clear(self):
        self.turns = []


# =========================================================================================== Manager
PLAN_SYSTEM = """You are the Management Consulting Manager Agent (engagement lead). You decompose client requests
into tasks for specialist agents, decide dependencies, and define success criteria. Reply with ONE JSON object."""

PLAN_PROMPT = """SPECIALIST AGENTS:
{agents}

ROUTER HINT (keyword/semantic classifier probabilities): {hint}

{memory}CLIENT REQUEST: "{q}"

Plan the engagement. Rules:
- Use the minimum set of agents that fully answers the request (usually 1; 2-3 only if it truly spans domains).
- Each task must be self-contained and specific (resolve references to earlier turns, name companies/models/years).
- Add a dependency only when an agent needs another agent's output (e.g. assess the financial risk of the vendor
  whose platform the Technology agent recommends). Dependencies must point to earlier steps.
Return JSON: {{"understanding": "<client need in one sentence>", "intent": "<troubleshooting | product_recommendation |
risk_analysis | financial_analysis | investment_strategy | research_summary | cross_sell>", "steps": [{{"id": "s1",
"agent": "<agent key>", "task": "...", "depends_on": []}}], "success_criteria": ["..."]}}"""

REVIEW_PROMPT = """CLIENT REQUEST: {q}
SUCCESS CRITERIA: {criteria}

AGENT RESULTS:
{results}

Do these results satisfy the success criteria? If an important gap can be closed by a specialist agent, issue at
most 2 follow-up tasks (agents: {agents}). Do not repeat work already done.
Return JSON: {{"sufficient": true|false, "gaps": ["..."], "followups": [{{"agent": "<key>", "task": "..."}}]}}"""

REVISE_PROMPT = """Revise the consulting recommendation below to fix the reviewer's issues. Keep the same section
structure and keep only citations that appear in the evidence. Do not add facts that are not in the evidence.

ISSUES: {issues}
REQUIRED CHANGES: {changes}

RECOMMENDATION:
{rec}"""


@dataclass
class AgenticResponse(ConsultingResponse):
    plan: Dict = field(default_factory=dict)
    trace: List[Dict] = field(default_factory=list)
    critique: Dict = field(default_factory=dict)
    review: Dict = field(default_factory=dict)
    llm_calls: int = 0

    def to_dict(self):
        d = super().to_dict()
        d.update(plan=self.plan, trace=self.trace, critique=self.critique, review=self.review, llm_calls=self.llm_calls)
        return d


class AgenticManager(ConsultingManager):
    def __init__(self, llm: Optional[LLM] = None, top_k: int = config.TOP_K, max_steps: int = 5,
                 enable_review: bool = True, enable_critic: bool = True, memory: Optional[ConversationMemory] = None):
        super().__init__(llm if llm is not None else get_llm(), top_k)
        self.max_steps, self.enable_review, self.enable_critic = max_steps, enable_review, enable_critic
        self.memory = memory or ConversationMemory()

    # ------------------------------------------------------------------------------- planning
    def plan(self, query: str, trace: Trace, force: Optional[List[str]] = None) -> Dict:
        hint = self.router.route_heuristic(query)
        fallback = {"understanding": query, "intent": hint.intent, "source": "router",
                    "steps": [{"id": f"s{i + 1}", "agent": d, "task": query, "depends_on": []}
                              for i, d in enumerate(force or hint.domains)],
                    "success_criteria": ["Answer the request with cited evidence", "Give prioritised next actions with KPIs"],
                    "router_probs": hint.probs}
        if force or not self.llm.available:
            trace.log("Manager", "plan", f"Plan ({'manual' if force else 'router'}): " +
                      ", ".join(s["agent"] for s in fallback["steps"]), fallback)
            return fallback
        agents = "\n".join(f"- {d}: {m['label']} | knowledge: {m['sources']} | tools: {', '.join(get_tools(d))}"
                           for d, m in config.DOMAINS.items())
        mem = f"CONVERSATION SO FAR:\n{self.memory.context()}\n\n" if self.memory.turns else ""
        try:
            p = _parse_json(self.llm.generate(
                PLAN_PROMPT.format(agents=agents, hint={k: v for k, v in hint.probs.items() if v > 0.05},
                                   memory=mem, q=query),
                system=PLAN_SYSTEM, temperature=0.0, max_tokens=1200, json_mode=True))
            steps, seen = [], set()
            for i, s in enumerate(p.get("steps", [])[:4]):
                if s.get("agent") not in AGENTS:
                    continue
                sid = str(s.get("id") or f"s{i + 1}")
                deps = [x for x in (s.get("depends_on") or []) if x in seen]
                steps.append({"id": sid, "agent": s["agent"], "task": str(s.get("task") or query), "depends_on": deps})
                seen.add(sid)
            if not steps:
                raise ValueError("empty plan")
            p["steps"], p["source"], p["router_probs"] = steps, "llm", hint.probs
            p.setdefault("success_criteria", fallback["success_criteria"])
            p.setdefault("intent", hint.intent)
            trace.log("Manager", "plan", _short(p.get("understanding", ""), 200),
                      {"steps": steps, "success_criteria": p["success_criteria"]})
            return p
        except Exception as e:
            trace.log("Manager", "fallback", f"Planner failed ({_short(e, 80)}); using router plan")
            trace.log("Manager", "plan", "Plan (router): " + ", ".join(s["agent"] for s in fallback["steps"]), fallback)
            return fallback

    # ------------------------------------------------------------------------------- execution
    @staticmethod
    def _summary(r: AgentResult, n: int = 1500) -> str:
        recs = "; ".join(f"{x['title']} ({x.get('rationale', '')[:60]})" for x in r.recommendations[:4])
        return (f"--- {r.label} (task: {r.task or r.question}) confidence {r.confidence.get('level')}\n"
                f"{_short(r.answer, n)}\nTop recommendations: {recs or 'n/a'}\n"
                f"Missing info: {'; '.join(r.missing_info[:4]) or 'none'}")

    def _run_step(self, step: Dict, query: str, done: Dict[str, AgentResult], trace: Trace) -> AgentResult:
        upstream = "\n\n".join(self._summary(done[d]) for d in step["depends_on"] if d in done)
        allowed = {e["cid"]: e.get("text", "") for d in step["depends_on"] if d in done for e in done[d].evidence}
        trace.log("Manager", "delegate", f"{config.DOMAINS[step['agent']]['label']}: {_short(step['task'], 160)}"
                  + (f"  (uses {', '.join(step['depends_on'])})" if step["depends_on"] else ""))
        agent = ReActAgent(step["agent"], self.llm, trace, max_steps=self.max_steps, top_k=self.top_k)
        return agent.run(step["task"], query, upstream=upstream, memory=self.memory.context(), allowed_ids=allowed)

    def execute(self, plan: Dict, query: str, trace: Trace) -> List[AgentResult]:
        steps = {s["id"]: s for s in plan["steps"]}
        done: Dict[str, AgentResult] = {}
        while len(done) < len(steps):
            wave = [s for sid, s in steps.items() if sid not in done and all(d in done for d in s["depends_on"])]
            if not wave:                                         # broken dependency graph -> run the rest
                wave = [s for sid, s in steps.items() if sid not in done]
                for s in wave:
                    s["depends_on"] = []
            with ThreadPoolExecutor(max_workers=len(wave)) as ex:
                outs = list(ex.map(lambda s: self._run_step(s, query, done, trace), wave))
            for s, r in zip(wave, outs):
                done[s["id"]] = r
        return [done[sid] for sid in steps]

    def review(self, query: str, plan: Dict, results: List[AgentResult], trace: Trace) -> Dict:
        if all(r.confidence.get("level") == "High" for r in results) and not any(r.error for r in results):
            out = {"sufficient": True, "gaps": [], "followups": [], "by": "rule: all agents high confidence"}
            trace.log("Manager", "review", "Results sufficient (all agents high confidence)")
            return out
        try:
            out = _parse_json(self.llm.generate(
                REVIEW_PROMPT.format(q=query, criteria=plan.get("success_criteria"),
                                     results="\n\n".join(self._summary(r, 1000) for r in results),
                                     agents=", ".join(AGENTS)),
                system=PLAN_SYSTEM, temperature=0.0, max_tokens=700, json_mode=True))
        except Exception as e:
            out = {"sufficient": True, "gaps": [f"review unavailable: {_short(e, 60)}"], "followups": []}
        out["followups"] = [f for f in (out.get("followups") or []) if isinstance(f, dict)
                            and f.get("agent") in AGENTS and f.get("task")][:2]
        trace.log("Manager", "review", "Sufficient" if out.get("sufficient") and not out["followups"]
                  else f"Gaps found -> {len(out['followups'])} follow-up task(s)", out)
        return out

    # ------------------------------------------------------------------------------- main
    def ask(self, query: str, strategy: str = "auto", force_domains: Optional[List[str]] = None,
            consolidate: bool = True, on_event=None,
            memory: Optional[ConversationMemory] = None) -> AgenticResponse:
        if memory is not None:
            self.memory = memory                     # per-session memory (UI keeps one per browser session)
        t0 = time.time()
        calls0 = self.llm.calls
        trace = Trace(on_event)
        if self.memory.turns:
            trace.log("Manager", "memory", f"Using {len(self.memory.turns)} previous turn(s) as context")
        if not self.llm.available:
            return self._ask_rule_based(query, strategy, force_domains, trace, t0)

        plan = self.plan(query, trace, force_domains)
        results = self.execute(plan, query, trace)
        review = {}
        if self.enable_review:
            review = self.review(query, plan, results, trace)
            if review.get("followups"):
                fsteps = [{"id": f"f{i + 1}", "agent": f["agent"], "task": f["task"],
                           "depends_on": []} for i, f in enumerate(review["followups"])]
                done = {f"r{i}": r for i, r in enumerate(results)}
                for fs in fsteps:
                    fs["depends_on"] = list(done)                 # follow-ups see everything so far
                    trace.log("Manager", "followup", f"{fs['agent']}: {_short(fs['task'], 150)}")
                with ThreadPoolExecutor(max_workers=len(fsteps)) as ex:
                    extra = list(ex.map(lambda s: self._run_step(s, query, done, trace), fsteps))
                for r in extra:
                    r.label += " (follow-up)"
                results += extra

        route = RouteDecision(domains=list(dict.fromkeys(r.domain for r in results)),
                              primary=results[0].domain, probs=plan.get("router_probs", {}),
                              intent=plan.get("intent", "research_summary"), strategy=f"planner ({plan.get('source')})",
                              reason=plan.get("understanding", ""), cross_domain=len({r.domain for r in results}) > 1)
        conf = self.overall_confidence(route, results)
        trace.log("Manager", "consolidate", "Consolidating findings into a client recommendation")
        rec = self.consolidate(query, route, results, conf) if consolidate else ""
        critique = {}
        if self.enable_critic and rec:
            critic = Critic(self.llm, trace)
            critique = critic.review(query, rec, results)
            if critique["verdict"] == "revise":
                try:
                    rec2 = self.llm.generate(REVISE_PROMPT.format(issues=critique["issues"],
                                                                  changes=critique.get("required_changes"), rec=rec),
                                             system="You are the Management Consulting Manager Agent.",
                                             max_tokens=2400)
                    ev = [e for r in results for e in r.evidence]
                    before = response_metrics(rec, ev, query)
                    after = response_metrics(rec2, ev, query)
                    if after["citation_validity"] >= before["citation_validity"] - 1e-9 and len(rec2) > 100:
                        rec = rec2
                        critique["revised"] = True
                        critique["after_metrics"] = after
                        trace.log("Manager", "revise", "Recommendation revised to address the critique",
                                  {"citation_validity": [before["citation_validity"], after["citation_validity"]]})
                    else:
                        trace.log("Manager", "revise", "Revision rejected (it reduced citation validity)")
                except LLMError as e:
                    trace.log("Manager", "fallback", f"Revision skipped: {_short(e, 80)}")
        trace.log("Manager", "final", f"Delivered ({self.llm.calls - calls0} LLM calls, {time.time() - t0:.1f}s)")
        resp = AgenticResponse(query=query, route=route, agents=results, recommendation=rec, confidence=conf,
                               llm=self.llm.label, latency_s=round(time.time() - t0, 2),
                               sub_questions={r.domain: r.task for r in results}, plan=plan,
                               trace=trace.to_list(), critique=critique, review=review,
                               llm_calls=self.llm.calls - calls0)
        self.memory.add(query, resp)
        return resp

    # ------------------------------------------------------------------------------- no-LLM mode
    def _ask_rule_based(self, query, strategy, force, trace, t0) -> AgenticResponse:
        plan = self.plan(query, trace, force)
        base = ConsultingManager.ask(self, query, strategy="hybrid" if strategy in ("auto", "llm") else strategy,
                                     force_domains=[s["agent"] for s in plan["steps"]])
        for r in base.agents:
            trace.log("Manager", "delegate", f"{r.label}: {_short(query, 120)}")
            trace.log(r.label, "tool", f"retrieve(top_k={self.top_k}) -> {len(r.evidence)} passages")
            if r.recommendations:
                trace.log(r.label, "tool", f"recommender -> {len(r.recommendations)} ranked items")
            if (r.extras or {}).get("hypotheses"):
                trace.log(r.label, "tool", f"diagnose_symptoms -> {len(r.extras['hypotheses'])} hypotheses")
            chk = self_check(r.answer, r.evidence, query)
            r.self_check = chk
            trace.log(r.label, "reflection", "Self-check passed" if chk["passed"] else "; ".join(chk["issues"]),
                      chk["metrics"])
            trace.log(r.label, "final", f"Answer ready (extractive, confidence {r.confidence.get('level')})")
        critique = Critic(self.llm, trace).review(query, base.recommendation, base.agents) if self.enable_critic else {}
        trace.log("Manager", "final", f"Delivered in rule-based mode ({time.time() - t0:.1f}s)")
        resp = AgenticResponse(query=query, route=base.route, agents=base.agents, recommendation=base.recommendation,
                               confidence=base.confidence, llm=self.llm.label, latency_s=round(time.time() - t0, 2),
                               sub_questions=base.sub_questions, plan=plan, trace=trace.to_list(), critique=critique)
        self.memory.add(query, resp)
        return resp
