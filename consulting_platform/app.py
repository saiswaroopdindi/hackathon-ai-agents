"""Streamlit UI for the Agentic AI Management Consulting Platform.   Run:  streamlit run app.py"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Agentic AI Consulting Platform", layout="wide")

EXAMPLES = {
    "Cisco router outage (Technology)": "Our Cisco ISR4331 enterprise router is experiencing intermittent connectivity failures. Which replacement parts should we order, and what actions should we take to prevent future outages?",
    "Lab instrument (Clinical)": "Which diagnostic instrument best meets our pathology laboratory requirements for fast whole-slide imaging?",
    "Company risks (SEC)": "What are the financial, competitive, and operational risks associated with Cisco and Agilent?",
    "Food insecurity 2023 (CSR)": "List major food insecurity reasons in 2023 from within the context of the SOFI 2023 report",
    "Malnutrition in war zones (CSR)": "Explain malnutrition in war zones",
    "Food prices (CSR)": "Explain the impact of increased food prices on food security",
    "Tech for food security (CSR)": "Which technology investments could improve food security and sustainability outcomes?",
    "Cross-domain: tech -> finance": "We are replacing our branch Cisco routers. Which platform should we move to, and what are Cisco's financial and supply-chain risks we should consider before signing a 5-year contract?",
    "Cross-domain: finance + CSR + tech": "Should our ESG fund invest in Cisco given its competitive risks, and which technology investments could improve food security outcomes?",
    "Checkout cross-sell (Retail)": "Customer added a Bosch D7050 Detector and a DSC PG9914 Motion Detector to the cart. What should we recommend?",
}
LEVEL_COLOR = {"High": "green", "Medium": "orange", "Low": "red"}
INDEXES = ("technology", "clinical", "sec", "csr", "trials")


# ------------------------------------------------------------------------------------------- setup
def index_ready() -> bool:
    return all((config.INDEX_DIR / f"{n}.pkl").exists() for n in INDEXES)


@st.cache_resource(show_spinner="Loading knowledge bases ...")
def get_manager(provider: str, top_k: int, _key_sig: str):
    from core.agentic import AgenticManager
    from core.llm import LLM
    from core.retriever import Retriever
    for n in INDEXES:
        Retriever(n)          # warm caches
    return AgenticManager(LLM(provider), top_k=top_k)


@st.cache_resource
def get_basket():
    from core.basket import BasketRecommender
    return BasketRecommender()


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM provider", ["auto", "gemini", "openai", "anthropic", "ollama", "none"],
                            help="'auto' uses the key in .env; 'none' = rule-based agents (no LLM)")
    key_env = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if key_env:
        k = st.text_input(f"{key_env}", type="password", value="", placeholder="optional - overrides .env")
        if k:
            os.environ[key_env] = k.strip()
    st.subheader("Agent behaviour")
    max_steps = st.slider("Max tool calls per agent", 2, 8, 4)
    enable_review = st.checkbox("Manager reviews results & issues follow-ups", True)
    enable_critic = st.checkbox("Critic agent reviews the final recommendation", True)
    force = st.multiselect("Force agents (skip planning)", list(config.DOMAINS.keys()))
    top_k = st.slider("Passages per search (top-k)", 3, 10, config.TOP_K)
    key_sig = "|".join(f"{e}:{bool(os.getenv(e))}" for e in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
                                                              "ANTHROPIC_API_KEY"))
    if not index_ready():
        with st.spinner("First run: building the knowledge-base indexes (about 2 minutes, only once) ..."):
            from core.retriever import build_all
            build_all([n for n in INDEXES if not (config.INDEX_DIR / f"{n}.pkl").exists()])
    if index_ready():
        mgr = get_manager(provider, top_k, key_sig)
        mgr.max_steps, mgr.enable_review, mgr.enable_critic = max_steps, enable_review, enable_critic
        st.success(f"LLM: {mgr.llm.label}")
        stats = json.loads((config.INDEX_DIR / "stats.json").read_text())
        st.caption("Knowledge base (chunks): " + ", ".join(f"{k} {v:,}" for k, v in stats.items()))
    else:
        mgr = None
        st.error(f"Could not build the indexes - is the hackathon data in {config.DATA_ROOT}?")
    if st.button("New conversation (clear memory)"):
        st.session_state.pop("history", None)
        st.session_state.pop("memory", None)
        st.rerun()

st.title("Agentic AI Management Consulting Platform")
st.caption("A Manager Agent plans the engagement and delegates tasks to specialist agents that reason with tools "
           "(search, diagnose, rank, compare); results are reviewed, consolidated and checked by a Critic agent.")

tab_chat, tab_cart, tab_eval, tab_arch = st.tabs(["Consulting Copilot", "Checkout Recommender",
                                                  "Evaluation", "Architecture"])


# ------------------------------------------------------------------------------------------- rendering
def render_trace(resp):
    by = {}
    for e in resp.trace:
        by.setdefault(e["actor"], []).append(e)
    order = ["Manager"] + [a for a in by if a not in ("Manager", "Critic")] + (["Critic"] if "Critic" in by else [])
    for actor in order:
        st.markdown(f"**{actor}**")
        for e in by.get(actor, []):
            line = f"`{e['t']:>5}s` {e['icon']} **{e['kind']}** - {e['title']}"
            st.markdown(line)
            if e["kind"] in ("plan", "review", "critique") and e.get("detail"):
                with st.popover("details"):
                    st.json(e["detail"])


def render_agent(a, expanded=False):
    lvl = a.confidence.get("level", "Low")
    head = (f"{a.label} - confidence :{LEVEL_COLOR.get(lvl, 'gray')}[{lvl} {a.confidence.get('score')}] - "
            f"{a.mode} - {a.tool_calls} tool calls - {a.revisions} revision(s) - {a.latency_s}s")
    with st.expander(head, expanded=expanded):
        if a.task:
            st.caption(f"Task from Manager: {a.task}")
        if a.error:
            st.error(a.error)
        st.markdown(a.answer)
        if a.steps:
            with st.popover("Reasoning steps (thought -> action -> observation)"):
                for i, s in enumerate(a.steps, 1):
                    st.markdown(f"**Step {i}** - *{s.get('thought', '')}*")
                    st.code(f"{s.get('action')}({json.dumps(s.get('args', {}))})\n-> {str(s.get('observation', ''))[:1200]}")
        hyps = (a.extras or {}).get("hypotheses")
        if hyps:
            st.markdown("**Fault hypotheses (not confirmed until diagnostics are run)**")
            st.dataframe(pd.DataFrame([{"hypothesis": h["hypothesis"], "likelihood": h["likelihood"],
                                        "status": h["status"], "diagnostics": " | ".join(h["diagnostics"])}
                                       for h in hyps]), use_container_width=True, hide_index=True)
        if a.recommendations:
            st.markdown("**Ranked recommendations (recommender tools)**")
            st.dataframe(pd.DataFrame(a.recommendations).drop(columns=["snippet", "datasheet_context"],
                                                               errors="ignore"),
                         use_container_width=True, hide_index=True)
        rm = (a.extras or {}).get("risk_matrix")
        if rm:
            st.markdown("**Risk matrix (risk-language mentions in 10-K passages)**")
            st.dataframe(pd.DataFrame(rm).fillna(0).astype(int), use_container_width=True)
        trials = (a.extras or {}).get("trials")
        if trials:
            st.markdown("**Related clinical studies (ClinicalTrials.gov)**")
            st.dataframe(pd.DataFrame(trials), use_container_width=True, hide_index=True)
        if a.next_actions:
            st.markdown("**Next actions**")
            st.dataframe(pd.DataFrame(a.next_actions), use_container_width=True, hide_index=True)
        if a.missing_info:
            st.markdown("**Information needed:** " + "; ".join(map(str, a.missing_info)))
        if a.self_check:
            sc = a.self_check
            st.caption(("Self-check passed" if sc.get("passed") else "Self-check issues: " + "; ".join(sc["issues"]))
                       + f" | metrics: {sc.get('metrics')}")
        st.caption("Confidence factors: " + "; ".join(a.confidence.get("reasons", [])))
        if a.evidence:
            st.markdown("**Evidence (citations)**")
            for e in a.evidence:
                link = f" - [link]({e['url']})" if e.get("url") else ""
                st.markdown(f"**[{e.get('cid')}]** {e.get('citation')} - score {e.get('score')}{link}")
                st.caption(" ".join(str(e.get("text", "")).split())[:450])


def render_response(resp):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Agents", len(resp.agents))
    m2.metric("Tool calls", sum(a.tool_calls for a in resp.agents))
    m3.metric("Confidence", f"{resp.confidence['level']} ({resp.confidence['score']})")
    m4.metric("Critic", (resp.critique or {}).get("verdict", "off").upper()
              + (" -> revised" if (resp.critique or {}).get("revised") else ""))
    m5.metric("Latency / LLM calls", f"{resp.latency_s}s / {getattr(resp, 'llm_calls', 0)}")

    plan = resp.plan or {}
    with st.expander("Manager plan", expanded=False):
        st.markdown(f"**Understanding:** {plan.get('understanding', resp.query)}  \n"
                    f"**Intent:** {plan.get('intent', resp.route.intent)} - planner: {plan.get('source', 'router')}")
        if plan.get("steps"):
            st.dataframe(pd.DataFrame(plan["steps"]), use_container_width=True, hide_index=True)
        if plan.get("success_criteria"):
            st.markdown("**Success criteria:** " + "; ".join(map(str, plan["success_criteria"])))
        if plan.get("router_probs"):
            st.bar_chart(pd.Series(plan["router_probs"], name="router hint"))
    with st.expander("Agent reasoning trace (plan -> tool calls -> self-checks -> review -> critique)"):
        render_trace(resp)

    st.markdown(resp.recommendation)
    crit = resp.critique or {}
    if crit:
        with st.expander(f"Critic review: {crit.get('verdict', '').upper()} ({crit.get('reviewer', '')})"):
            if crit.get("scores"):
                st.write(crit["scores"])
            for i in crit.get("issues", []):
                st.markdown(f"- {i}")
    st.markdown("##### Specialist agent findings")
    for a in resp.agents:
        render_agent(a, expanded=False)
    st.download_button("Download full analysis + trace (JSON)", json.dumps(resp.to_dict(), indent=2, default=str),
                       file_name="consulting_analysis.json", key=f"dl_{id(resp)}")


# ------------------------------------------------------------------------------------------- copilot (chat)
with tab_chat:
    from core.agentic import ConversationMemory
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("memory", ConversationMemory())

    ex = st.selectbox("Try an example request", ["(type your own below)"] + list(EXAMPLES.keys()))
    run_example = st.button("Run example", disabled=mgr is None or ex.startswith("("))

    for q_old, r_old in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q_old)
        with st.chat_message("assistant"):
            render_response(r_old)

    q = st.chat_input("Describe the client's business problem ...", disabled=mgr is None)
    if run_example:
        q = EXAMPLES[ex]
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.status("Manager is planning and delegating to specialist agents ...", expanded=False) as status:
                resp = mgr.ask(q, force_domains=force or None, memory=st.session_state.memory)
                status.update(label=f"Done: {len(resp.agents)} agent(s), "
                                    f"{sum(a.tool_calls for a in resp.agents)} tool calls, {resp.latency_s}s",
                              state="complete")
            render_response(resp)
        st.session_state.history.append((q, resp))

# ------------------------------------------------------------------------------------------- checkout
with tab_cart:
    st.subheader("Electronic Item Checkout Portal - market basket recommendations")
    bm = get_basket()
    st.session_state.setdefault("cart", [])
    cc1, cc2 = st.columns([2, 3])
    with cc1:
        choice = st.selectbox("Select item from the list", [i for i in bm.items if i not in st.session_state.cart])
        b1, b2 = st.columns(2)
        if b1.button("Add item"):
            st.session_state.cart.append(choice)
            st.rerun()
        if b2.button("Clear cart"):
            st.session_state.cart = []
            st.rerun()
        st.markdown("**Added items to cart:**")
        for i, it in enumerate(st.session_state.cart, 1):
            st.write(f"{i}. {it}")
    with cc2:
        cart = st.session_state.cart
        if cart:
            from core.basket import exact_cart_seen
            recs = bm.recommend(cart, k=5)
            seen = exact_cart_seen(bm.tx, cart)
            st.markdown(f"**Recommended items** (cart combination seen in {seen} of {bm.n} historical baskets)")
            if not recs:
                st.info("No reliable co-purchase pattern for this combination - no recommendations shown "
                        "(avoids irrelevant suggestions).")
            for j, rec in enumerate(recs):
                c_a, c_b = st.columns([4, 1])
                c_a.markdown(f"**{rec.item}**  \nconfidence {rec.confidence:.0%} - lift {rec.lift:.2f} - "
                             f"seen {rec.support_count}x with *{', '.join(rec.antecedent)}*")
                if c_b.button("Add", key=f"add_{j}"):
                    st.session_state.cart.append(rec.item)
                    st.rerun()
        else:
            st.info("Cart is empty - add the first item to get recommendations.")
    with st.expander("Strongest association rules in the transaction history"):
        st.dataframe(pd.DataFrame(bm.top_rules(15)), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------------------- evaluation
with tab_eval:
    rep = config.EVAL_DIR / "results" / "report.md"
    if st.button("Run offline evaluation (routing, retrieval, recommenders, responses)", disabled=mgr is None):
        import evaluate as ev
        from core.llm import LLM
        with st.spinner("Evaluating ..."):
            res = {"llm": "No LLM (extractive mode)", "routing": ev.eval_routing(None),
                   "routing_holdout": ev.eval_routing(None, "routing_holdout.json"),
                   "retrieval": ev.eval_retrieval(), "recommenders": ev.eval_recommenders(),
                   "responses": ev.eval_responses(LLM("none"))}
            prev = config.EVAL_DIR / "results" / "results.json"
            if prev.exists():                       # keep agentic results from `python evaluate.py --with-llm`
                old = json.loads(prev.read_text())
                for key in ("agentic", "routing_planner"):
                    if key in old:
                        res[key] = old[key]
            ev.write_report(res)
    st.caption("Agent-behaviour metrics (planner routing, tool use, self-check, critic) need an LLM: "
               "run `python evaluate.py --with-llm` in a terminal.")
    if rep.exists():
        st.markdown(rep.read_text())
        rj = config.EVAL_DIR / "results" / "results.json"
        if rj.exists():
            with st.expander("Raw results JSON"):
                st.json(json.loads(rj.read_text()))
    else:
        st.info("No evaluation yet - click the button or run `python evaluate.py`.")

# ------------------------------------------------------------------------------------------- architecture
with tab_arch:
    st.graphviz_chart("""
digraph G { rankdir=TB; compound=true; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#f1f5f9"];
  U [label="Client request (chat, multi-turn)", fillcolor="#e0e7ff"];
  MEM [label="Conversation memory", fillcolor="#e0f2fe"];
  subgraph cluster_m { label="Management Consulting Manager Agent"; style=rounded; color="#6366f1";
    P [label="1. Plan\\nagents, self-contained tasks,\\ndependencies, success criteria", fillcolor="#c7d2fe"];
    D [label="2. Delegate\\nparallel waves; dependent tasks\\nreceive upstream findings", fillcolor="#c7d2fe"];
    RV [label="3. Review\\ncriteria met? -> follow-up tasks", fillcolor="#c7d2fe"];
    CO [label="4. Consolidate", fillcolor="#c7d2fe"]; }
  subgraph cluster_a { label="Specialist agents (ReAct: thought -> tool -> observation -> self-check)"; style=rounded; color="#16a34a";
    T [label="Technology\\nsearch_tech_docs, identify_router_model,\\ndiagnose_symptoms, rank_components,\\nrecommend_platforms"];
    C [label="Clinical\\nsearch_clinical_literature,\\nrecommend_agilent_products,\\nsearch_clinical_trials, trial_landscape"];
    S [label="SEC\\nlist_filings, search_filings,\\nrisk_matrix, compare_companies"];
    E [label="Corporate Responsibility\\nsearch_sofi, rank_investments"];
    R [label="Retail\\nidentify_cart_items,\\nrecommend_addons, top_associations"]; }
  K [label="Knowledge bases: BM25 hybrid retrieval with page-level citations\\nCisco/Google docs - clinical PDFs + 32k trials - 10-Ks - SOFI 2023/24 - 1,000 baskets", fillcolor="#fef3c7"];
  CR [label="5. Critic agent\\ncitations valid? grounded? hypotheses labelled?\\nKPIs? -> approve / revise", fillcolor="#fecaca"];
  O [label="Client recommendation\\nexecutive summary - findings - P1-P3 actions with KPIs - confidence", fillcolor="#bbf7d0"];
  U -> P; MEM -> P; P -> D; D -> T; D -> C; D -> S; D -> E; D -> R;
  T -> K; C -> K; S -> K; E -> K; R -> K;
  T -> RV; C -> RV; S -> RV; E -> RV; R -> RV; RV -> D [style=dashed, label="follow-ups"];
  RV -> CO; CO -> CR; CR -> CO [style=dashed, label="revise"]; CR -> O; O -> MEM [style=dotted];
}""")
    st.markdown("""
**What makes it agentic**
- **Planning:** the Manager turns a request into tasks with dependencies and success criteria (not a fixed route).
- **Tool use:** each specialist decides which tools to call, with what arguments, and when it has enough evidence.
- **Collaboration:** dependent tasks receive upstream agents' findings (e.g. Technology's platform choice -> SEC vendor risk).
- **Self-correction:** agents self-check citations, grounding, KPIs and hypothesis labelling, and revise or search again.
- **Oversight:** the Manager reviews results against its success criteria and issues follow-up tasks; a Critic agent
  reviews the final recommendation and triggers a revision.
- **Memory:** follow-up questions ("what about Agilent?") are resolved using earlier turns.
- **Graceful degradation:** malformed outputs are retried, unknown tools are corrected, and any failure falls back to the
  deterministic pipeline, so the client always gets an answer.
""")
