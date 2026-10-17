"""Evaluation harness.

  python evaluate.py              # all offline metrics (no LLM needed)
  python evaluate.py --with-llm   # + LLM routing and LLM-generated answers / LLM-as-judge

Measures
  1. Routing accuracy      : primary accuracy, exact-set match, Jaccard, per-domain F1 (lexicon / prototype / hybrid / llm)
  2. Retrieval accuracy    : Hit@1, Hit@3, Hit@k, MRR@k, source precision@k (via each agent's retrieval pipeline)
  3. Recommendation quality: market basket Hit@5 vs popularity & random; trial recommender P@5 / nDCG@5;
                             CSR investment recommender Hit@1/3; fault-diagnosis Hit@1
  4. Response quality      : citation coverage, citation validity, lexical grounding, query coverage, actionability
                             (+ LLM-as-judge 1-5 when --with-llm)
Writes eval/results/results.json and eval/results/report.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict

import config
from core import basket
from core.agents import AGENTS, CSRAgent, TechnologyAgent
from core.llm import LLM, get_llm
from core.retriever import Retriever
from core.router import DOMAINS, Router
from core.text import split_sentences, tokenize

OUT = config.EVAL_DIR / "results"


# ----------------------------------------------------------------------------- 1. routing
def eval_routing(llm=None, fname="routing_testset.json"):
    data = json.loads((config.EVAL_DIR / fname).read_text())
    router = Router(llm)
    strategies = ["lexicon", "prototype", "hybrid"] + (["llm"] if llm and llm.available else [])
    res = {}
    for s in strategies:
        prim = exact = 0
        jac = []
        tp, fp, fn = Counter(), Counter(), Counter()
        conf = defaultdict(Counter)
        errors = []
        for ex in data:
            gold = ex["domains"]
            if s == "llm":
                try:
                    r = router.route_llm(ex["q"], router.route_heuristic(ex["q"]))
                except Exception as e:
                    r = router.route_heuristic(ex["q"])
            else:
                r = router.route_heuristic(ex["q"], s)
            pred = r.domains
            prim += pred[0] == gold[0] or (len(gold) > 1 and pred[0] in gold)
            exact += set(pred) == set(gold)
            jac.append(len(set(pred) & set(gold)) / len(set(pred) | set(gold)))
            conf[gold[0]][pred[0]] += 1
            for d in DOMAINS:
                tp[d] += d in pred and d in gold
                fp[d] += d in pred and d not in gold
                fn[d] += d not in pred and d in gold
            if set(pred) != set(gold):
                errors.append({"q": ex["q"], "gold": gold, "pred": pred})
        f1 = {}
        for d in DOMAINS:
            p = tp[d] / max(tp[d] + fp[d], 1)
            rc = tp[d] / max(tp[d] + fn[d], 1)
            f1[d] = round(2 * p * rc / max(p + rc, 1e-9), 3)
        n = len(data)
        cross = [ex for ex in data if len(ex["domains"]) > 1]
        res[s] = {"n": n, "primary_accuracy": round(prim / n, 3), "exact_set_match": round(exact / n, 3),
                  "mean_jaccard": round(sum(jac) / n, 3), "macro_f1": round(sum(f1.values()) / len(f1), 3),
                  "per_domain_f1": f1, "cross_domain_cases": len(cross),
                  "confusion_primary": {g: dict(c) for g, c in conf.items()}, "errors": errors[:12]}
    return res


# ----------------------------------------------------------------------------- 2. retrieval
def eval_retrieval(k=config.TOP_K):
    data = json.loads((config.EVAL_DIR / "retrieval_testset.json").read_text())
    per_dom = defaultdict(list)
    rows = []
    for ex in data:
        agent = AGENTS[ex["domain"]](None, top_k=k)
        hits = agent.retrieve(ex["q"])

        def rel(h):
            d = h.doc
            meta = (d.get("path", "") + " " + d.get("source", "") + " " + d.get("url", "")).lower()
            return any(s in meta for s in ex["source"]) and any(kw in d["text"].lower() for kw in ex["keywords"])

        def src_ok(h):
            d = h.doc
            meta = (d.get("path", "") + " " + d.get("source", "") + " " + d.get("url", "")).lower()
            return any(s in meta for s in ex["source"])

        flags = [rel(h) for h in hits]
        first = next((i for i, f in enumerate(flags) if f), None)
        row = {"domain": ex["domain"], "q": ex["q"], "hit@1": int(bool(flags[:1] and flags[0])),
               "hit@3": int(any(flags[:3])), f"hit@{k}": int(any(flags)),
               "rr": 0 if first is None else 1 / (first + 1),
               "source_precision": sum(src_ok(h) for h in hits) / max(len(hits), 1),
               "top": hits[0].citation if hits else ""}
        rows.append(row)
        per_dom[ex["domain"]].append(row)

    def agg(rs):
        return {"n": len(rs), "hit@1": round(statistics.mean(r["hit@1"] for r in rs), 3),
                "hit@3": round(statistics.mean(r["hit@3"] for r in rs), 3),
                f"hit@{k}": round(statistics.mean(r[f"hit@{k}"] for r in rs), 3),
                f"mrr@{k}": round(statistics.mean(r["rr"] for r in rs), 3),
                "source_precision@k": round(statistics.mean(r["source_precision"] for r in rs), 3)}
    return {"overall": agg(rows), "by_domain": {d: agg(rs) for d, rs in per_dom.items()},
            "misses": [r for r in rows if not r[f"hit@{k}"]]}


# ----------------------------------------------------------------------------- 3. recommenders
TRIAL_QUERIES = [
    ("insulin pump closed loop type 1 diabetes", ["type 1", "type1", "t1d"]),
    ("metformin type 2 diabetes", ["type 2", "type2", "t2d"]),
    ("COVID-19 vaccine immunogenicity", ["covid", "sars-cov-2"]),
    ("long COVID rehabilitation", ["covid", "post-acute"]),
    ("anterior cruciate ligament reconstruction rehabilitation", ["anterior cruciate", "acl", "knee"]),
    ("total knee arthroplasty pain management", ["knee", "arthroplasty"]),
    ("acute myocardial infarction stent", ["myocardial", "infarction", "coronary", "heart attack", "stemi"]),
    ("cardiac rehabilitation after heart attack", ["myocardial", "heart", "coronary", "cardiac"]),
]

CSR_REC_CASES = [
    ("help smallholder farmers sell crops with digital tools", "Digital agriculture"),
    ("reduce child stunting and anaemia in women", "Nutrition programmes"),
    ("hunger in conflict-affected regions and crises", "Humanitarian"),
    ("protect crops from droughts and climate shocks", "Climate-resilient"),
    ("too much food is lost in storage and transport", "Food loss"),
    ("poor families cannot afford food, cash support", "Social protection"),
    ("funding gap and financing to end hunger", "Financing"),
    ("cities and urban diets becoming unhealthy and expensive", "Peri-urban"),
]

FAULT_CASES = [
    ("router interface flapping with CRC errors on the fiber port", "Physical layer"),
    ("router randomly reboots after power events in the building", "Power"),
    ("fan alarm and overheat warnings on the router", "Thermal"),
    ("high cpu and memory leak after software upgrade", "Software"),
    ("BGP neighbor to the ISP keeps going down", "WAN"),
    ("throughput is saturating the router during peak hours", "Capacity"),
    ("intermittent connectivity drops on the LAN port", "Physical layer"),
    ("IPsec VPN tunnel to branch drops", "WAN"),
]


def eval_recommenders():
    out = {"market_basket": basket.evaluate(k=5)}
    # trial recommender: precision@5 and nDCG@5 against condition labels
    tr = Retriever("trials")
    p5, nd = [], []
    for q, cond in TRIAL_QUERIES:
        hits = tr.search(q, k=5, per_source=1)
        rels = [int(any(c in (h.doc["conditions"] + " " + h.doc["text"][:200]).lower() for c in cond)) for h in hits]
        p5.append(sum(rels) / 5)
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
        idcg = sum(1 / math.log2(i + 2) for i in range(5))
        nd.append(dcg / idcg)
    out["clinical_trial_recommender"] = {"queries": len(TRIAL_QUERIES), "precision@5": round(statistics.mean(p5), 3),
                                         "ndcg@5": round(statistics.mean(nd), 3)}
    # CSR investment recommender
    agent = CSRAgent(None)
    h1 = h3 = 0
    for q, exp in CSR_REC_CASES:
        recs = agent.specialist("Which investment should we fund to " + q, [])["recommendations"]
        titles = [r["title"] for r in recs]
        h1 += bool(titles) and exp.lower() in titles[0].lower()
        h3 += any(exp.lower() in t.lower() for t in titles[:3])
    out["csr_investment_recommender"] = {"cases": len(CSR_REC_CASES), "hit@1": round(h1 / len(CSR_REC_CASES), 3),
                                         "hit@3": round(h3 / len(CSR_REC_CASES), 3)}
    # technology fault-hypothesis ranking
    ta = TechnologyAgent(None)
    f1 = f2 = 0
    for q, exp in FAULT_CASES:
        hyps = ta.specialist(q, [])["extras"].get("hypotheses", [])
        names = [h["hypothesis"] for h in hyps]
        f1 += bool(names) and exp.lower() in names[0].lower()
        f2 += any(exp.lower() in n.lower() for n in names[:2])
    out["fault_diagnosis"] = {"cases": len(FAULT_CASES), "hit@1": round(f1 / len(FAULT_CASES), 3),
                              "hit@2": round(f2 / len(FAULT_CASES), 3)}
    return out


# ----------------------------------------------------------------------------- 4. response quality
DEMO_QUERIES = [
    "Our Cisco enterprise router is experiencing intermittent connectivity failures. Which replacement parts should we order, and what actions should we take to prevent future outages?",
    "Which diagnostic instrument best meets our pathology laboratory requirements for whole-slide imaging?",
    "What are the financial, competitive, and operational risks associated with Cisco and Agilent?",
    "List major food insecurity reasons in 2023 from within the context of the SOFI 2023 report",
    "Explain malnutrition in war zones",
    "Explain the impact of increased food prices on food security",
    "Which technology investments could improve food security and sustainability outcomes?",
    "Summarize clinical evidence on machine learning for heart attack prediction",
    "We are replacing our Cisco routers; what are Cisco's financial and supply-chain risks we should consider before signing a 5-year contract?",
]

from core.quality import CITE_RE, response_metrics  # noqa: E402


JUDGE_PROMPT = """Rate this consulting answer on 1-5 scales. Question: {q}
Answer:
{a}
Return JSON {{"relevance": n, "groundedness": n, "actionability": n, "clarity": n, "comment": "<short>"}}"""


def eval_responses(llm):
    from core.manager import ConsultingManager
    mgr = ConsultingManager(llm)
    rows = []
    for q in DEMO_QUERIES:
        t = time.time()
        r = mgr.ask(q)
        for a in r.agents:
            if a.domain == "retail":
                continue
            m = response_metrics(a.answer, a.evidence, q)
            m.update({"q": q[:80], "agent": a.domain, "mode": a.mode, "confidence": a.confidence.get("level"),
                      "n_recommendations": len(a.recommendations),
                      "actionable": int(bool(a.next_actions) and all(x.get("kpi") for x in a.next_actions))})
            rows.append(m)
        rec_m = response_metrics(r.recommendation, [e for a in r.agents for e in a.evidence], q)
        row = rows[-1]
        row["final_latency_s"] = round(time.time() - t, 2)
        if llm.available:
            try:
                j = llm.generate_json(JUDGE_PROMPT.format(q=q, a=r.recommendation[:6000]))
                row["judge"] = {k: j.get(k) for k in ("relevance", "groundedness", "actionability", "clarity")}
            except Exception as e:
                row["judge"] = {"error": str(e)[:80]}
        row["final_citation_validity"] = rec_m["citation_validity"]
    keys = ["citation_coverage", "citation_validity", "grounding_rate", "query_coverage", "actionable"]
    summary = {k: round(statistics.mean(r[k] for r in rows), 3) for k in keys}
    judges = [r["judge"] for r in rows if isinstance(r.get("judge"), dict) and "relevance" in r["judge"]]
    if judges:
        summary["llm_judge"] = {k: round(statistics.mean(float(j[k]) for j in judges if j.get(k) is not None), 2)
                                for k in ("relevance", "groundedness", "actionability", "clarity")}
    return {"summary": summary, "mode": llm.label, "rows": rows}


# ----------------------------------------------------------------------------- 5. agent behaviour (needs LLM)
def eval_planner(llm, fname="routing_testset.json", limit=None):
    """Routing accuracy of the Manager's LLM planner (which agents it decides to involve)."""
    from core.agentic import AgenticManager
    from core.trace import Trace
    data = json.loads((config.EVAL_DIR / fname).read_text())[:limit]
    mgr = AgenticManager(llm)
    prim = exact = 0
    jac, errors = [], []
    for ex in data:
        p = mgr.plan(ex["q"], Trace())
        pred = list(dict.fromkeys(s["agent"] for s in p["steps"]))
        gold = ex["domains"]
        prim += pred[0] in gold
        exact += set(pred) == set(gold)
        jac.append(len(set(pred) & set(gold)) / len(set(pred) | set(gold)))
        if set(pred) != set(gold):
            errors.append({"q": ex["q"], "gold": gold, "pred": pred, "source": p.get("source")})
    n = len(data)
    return {"n": n, "primary_accuracy": round(prim / n, 3), "exact_set_match": round(exact / n, 3),
            "mean_jaccard": round(sum(jac) / n, 3), "errors": errors[:12]}


def eval_agentic(llm, n_queries=5):
    """Run the full agentic workflow and measure how the agents behave, not just what they answer."""
    from core.agentic import AgenticManager
    from core.tools import get_tools
    rows, agents = [], []
    for q in DEMO_QUERIES[:n_queries]:
        mgr = AgenticManager(llm)
        r = mgr.ask(q)
        ev = [e for a in r.agents for e in a.evidence]
        m = response_metrics(r.recommendation, ev, q)
        crit = r.critique or {}
        rows.append({"q": q[:70], "agents": [a.domain for a in r.agents], "llm_calls": r.llm_calls,
                     "latency_s": r.latency_s, "followups": len((r.review or {}).get("followups", [])),
                     "critic_verdict": crit.get("verdict"), "critic_revised": bool(crit.get("revised")),
                     "critic_scores": crit.get("scores", {}), "final_citation_validity": m["citation_validity"],
                     "final_grounding": m["grounding_rate"], "planner": (r.plan or {}).get("source")})
        for a in r.agents:
            agents.append({"agent": a.domain, "mode": a.mode, "tool_calls": a.tool_calls, "tool_errors": a.tool_errors,
                           "distinct_tools": len({s.get("action") for s in a.steps if s.get("action") in get_tools(a.domain)}),
                           "revisions": a.revisions, "self_check_passed": bool((a.self_check or {}).get("passed")),
                           "confidence": a.confidence.get("level")})
    na = max(len(agents), 1)
    scores = [r["critic_scores"] for r in rows if r["critic_scores"]]
    summary = {
        "requests": len(rows), "agent_runs": len(agents),
        "completed_agentically": round(sum(a["mode"] == "agentic" for a in agents) / na, 3),
        "avg_tool_calls_per_agent": round(sum(a["tool_calls"] for a in agents) / na, 2),
        "avg_distinct_tools_per_agent": round(sum(a["distinct_tools"] for a in agents) / na, 2),
        "tool_error_rate": round(sum(a["tool_errors"] for a in agents) / max(sum(a["tool_calls"] for a in agents), 1), 3),
        "self_revision_rate": round(sum(a["revisions"] > 0 for a in agents) / na, 3),
        "final_self_check_pass_rate": round(sum(a["self_check_passed"] for a in agents) / na, 3),
        "planner_success_rate": round(sum(r["planner"] == "llm" for r in rows) / max(len(rows), 1), 3),
        "followup_rate": round(sum(r["followups"] > 0 for r in rows) / max(len(rows), 1), 3),
        "critic_approval_rate": round(sum(r["critic_verdict"] == "approve" for r in rows) / max(len(rows), 1), 3),
        "critic_revision_rate": round(sum(r["critic_revised"] for r in rows) / max(len(rows), 1), 3),
        "final_citation_validity": round(statistics.mean(r["final_citation_validity"] for r in rows), 3) if rows else 0,
        "avg_llm_calls_per_request": round(statistics.mean(r["llm_calls"] for r in rows), 1) if rows else 0,
        "avg_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 1) if rows else 0,
    }
    if scores:
        summary["critic_scores_avg"] = {k: round(statistics.mean(float(s[k]) for s in scores if s.get(k) is not None), 2)
                                        for k in ("relevance", "groundedness", "actionability", "risk_awareness")
                                        if any(s.get(k) is not None for s in scores)}
    return {"summary": summary, "requests": rows, "agents": agents, "llm": llm.label}


# ----------------------------------------------------------------------------- report
def write_report(res):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    L = ["# Evaluation report", f"_Generated {time.strftime('%Y-%m-%d %H:%M')} | LLM: {res.get('llm')}_", ""]
    L += ["## 1. Routing accuracy", f"Test set: {next(iter(res['routing'].values()))['n']} labelled requests "
          f"({next(iter(res['routing'].values()))['cross_domain_cases']} cross-domain).", "",
          "| Strategy | Primary acc. | Exact set match | Mean Jaccard | Macro F1 |", "|---|---|---|---|---|"]
    for s, m in res["routing"].items():
        L.append(f"| {s} | {m['primary_accuracy']:.1%} | {m['exact_set_match']:.1%} | {m['mean_jaccard']:.3f} | {m['macro_f1']:.3f} |")
    if "routing_holdout" in res:
        h = res["routing_holdout"]
        L += ["", f"**Hold-out set** ({next(iter(h.values()))['n']} requests written after the router was tuned, "
                  f"{next(iter(h.values()))['cross_domain_cases']} cross-domain):", "",
              "| Strategy | Primary acc. | Exact set match | Mean Jaccard | Macro F1 |", "|---|---|---|---|---|"]
        for s, m in h.items():
            L.append(f"| {s} | {m['primary_accuracy']:.1%} | {m['exact_set_match']:.1%} | {m['mean_jaccard']:.3f} | {m['macro_f1']:.3f} |")
    L += ["", "## 2. Retrieval accuracy (agent retrieval pipelines)", "",
          "| Domain | n | Hit@1 | Hit@3 | Hit@k | MRR@k | Source precision@k |", "|---|---|---|---|---|---|---|"]
    k = config.TOP_K
    for d, m in list(res["retrieval"]["by_domain"].items()) + [("**overall**", res["retrieval"]["overall"])]:
        L.append(f"| {d} | {m['n']} | {m['hit@1']:.1%} | {m['hit@3']:.1%} | {m[f'hit@{k}']:.1%} | "
                 f"{m[f'mrr@{k}']:.3f} | {m['source_precision@k']:.1%} |")
    rc = res["recommenders"]
    mb = rc["market_basket"]
    L += ["", "## 3. Recommendation relevance", "",
          f"**Market basket (leave-one-out, {mb['splits']} random 80/20 splits):** Hit@5 = {mb['hit_rate@5']:.1%} "
          f"(+/-{mb['hit_rate_std']:.1%}), popularity baseline = {mb['popularity_hit_rate@5']:.1%}, random = "
          f"{mb['random_expected_hit_rate@5']:.1%}, MRR = {mb['mrr']}, precision@5 = {mb['precision@5(answered)']}.",
          "", f"**Clinical trial recommender:** P@5 = {rc['clinical_trial_recommender']['precision@5']:.1%}, "
              f"nDCG@5 = {rc['clinical_trial_recommender']['ndcg@5']:.3f}",
          f"**CSR investment recommender:** Hit@1 = {rc['csr_investment_recommender']['hit@1']:.1%}, "
          f"Hit@3 = {rc['csr_investment_recommender']['hit@3']:.1%}",
          f"**Technology fault diagnosis:** Hit@1 = {rc['fault_diagnosis']['hit@1']:.1%}, "
          f"Hit@2 = {rc['fault_diagnosis']['hit@2']:.1%}"]
    if "responses" in res:
        s = res["responses"]["summary"]
        L += ["", f"## 4. Response quality ({res['responses']['mode']})", "",
              "| Citation coverage | Citation validity | Grounding rate | Query coverage | Actionable |",
              "|---|---|---|---|---|",
              f"| {s['citation_coverage']:.1%} | {s['citation_validity']:.1%} | {s['grounding_rate']:.1%} | "
              f"{s['query_coverage']:.1%} | {s['actionable']:.1%} |"]
        if "llm_judge" in s:
            L.append("\nLLM-as-judge (1-5): " + ", ".join(f"{k} {v}" for k, v in s["llm_judge"].items()))
    if "routing_planner" in res:
        L += ["", "## 5. Agentic behaviour", "", "**Manager planner routing (LLM):**", "",
              "| Test set | n | Primary acc. | Exact set match | Mean Jaccard |", "|---|---|---|---|---|"]
        for name, m in res["routing_planner"].items():
            L.append(f"| {name} | {m['n']} | {m['primary_accuracy']:.1%} | {m['exact_set_match']:.1%} | {m['mean_jaccard']:.3f} |")
    if "agentic" in res:
        s = res["agentic"]["summary"]
        L += ["", f"**End-to-end agent runs** ({s['requests']} requests, {s['agent_runs']} agent runs, "
                  f"{res['agentic']['llm']}):", "", "| Metric | Value |", "|---|---|"]
        labels = {"completed_agentically": "Agent runs completed by the ReAct loop (no fallback)",
                  "avg_tool_calls_per_agent": "Avg tool calls per agent", "avg_distinct_tools_per_agent":
                  "Avg distinct tools per agent", "tool_error_rate": "Tool error rate",
                  "self_revision_rate": "Agents that revised after self-check",
                  "final_self_check_pass_rate": "Final self-check pass rate",
                  "planner_success_rate": "Planner produced a valid plan", "followup_rate":
                  "Requests with Manager follow-up tasks", "critic_approval_rate": "Critic approved first draft",
                  "critic_revision_rate": "Recommendations revised after critique",
                  "final_citation_validity": "Citation validity of final recommendation",
                  "avg_llm_calls_per_request": "Avg LLM calls per request", "avg_latency_s": "Avg latency (s)"}
        for k, lab in labels.items():
            v = s.get(k)
            L.append(f"| {lab} | {v:.1%} |" if isinstance(v, float) and v <= 1 and k not in
                     ("avg_tool_calls_per_agent", "avg_distinct_tools_per_agent", "avg_llm_calls_per_request",
                      "avg_latency_s") else f"| {lab} | {v} |")
        if s.get("critic_scores_avg"):
            L.append("\nCritic scores (1-5): " + ", ".join(f"{k} {v}" for k, v in s["critic_scores_avg"].items()))
    L += ["", "### Caveats (read before quoting numbers)",
          "- Test sets are small and were written by the project team; the CSR-investment and fault-diagnosis cases "
          "(8 each) are sanity checks, not benchmarks. The routing hold-out set is the fairest routing number.",
          "- Market basket: Hit@5 is close to both popularity and random baselines, i.e. the 1,000 transactions contain "
          "only weak co-purchase structure. The recommender therefore reports lift/support with each suggestion and "
          "abstains when no rule meets the support and lift thresholds.",
          "- Without an LLM, answers are extractive (evidence sentences), so grounding is high by construction; re-run "
          "with `--with-llm` to measure generated answers.",
          "", "### Metric definitions",
          "- *Primary acc.*: top routed domain is (one of) the gold domains; *Exact set match*: routed set == gold set.",
          "- *Hit@k*: a passage from the expected source containing an expected answer phrase is in the top-k.",
          "- *Citation validity*: cited ids exist in the evidence; *Grounding rate*: cited sentences share >=30% of "
          "their content words with the cited passage.",
          "- *Market basket Hit@5*: hide one item of each held-out basket, recommend from the rest."]
    (OUT / "report.md").write_text("\n".join(L))
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--skip-responses", action="store_true")
    ap.add_argument("--agent-queries", type=int, default=5, help="end-to-end agentic runs (LLM mode)")
    ap.add_argument("--planner-limit", type=int, default=None, help="limit planner routing cases (saves quota)")
    a = ap.parse_args()
    llm = get_llm() if a.with_llm else LLM("none")
    if a.with_llm and not llm.available:
        print("No LLM configured (.env) - running offline metrics only.")
    t = time.time()
    res = {"llm": llm.label}
    print("1/5 routing ...");      res["routing"] = eval_routing(None)
    res["routing_holdout"] = eval_routing(None, "routing_holdout.json")
    print("2/5 retrieval ...");    res["retrieval"] = eval_retrieval()
    print("3/5 recommenders ..."); res["recommenders"] = eval_recommenders()
    if not a.skip_responses:
        print("4/5 responses (pipeline, extractive) ...");  res["responses"] = eval_responses(LLM("none"))
    if llm.available:
        print("5/5 agentic behaviour (LLM) ...")
        res["routing_planner"] = {"main": eval_planner(llm, "routing_testset.json", a.planner_limit),
                                  "holdout": eval_planner(llm, "routing_holdout.json", a.planner_limit)}
        res["agentic"] = eval_agentic(llm, a.agent_queries)
    print(write_report(res))
    print(f"\nSaved to {OUT}  ({time.time()-t:.0f}s)")
