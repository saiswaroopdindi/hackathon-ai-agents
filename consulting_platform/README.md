# Agentic AI Management Consulting Platform
CMPE 256 Hackathon (SJSU, Fall 2026): a multi-agent "AI consulting coworker".

The core of the platform is the agents. A **Manager Agent** plans each engagement, delegates tasks to **specialist
agents**, reviews their work and issues follow-ups. Each specialist **reasons with tools**: it searches, diagnoses,
ranks and compares, and it checks its own answer. A **Critic agent** reviews the final recommendation before the
client sees it.

```
            ┌──────────────────────── Management Consulting Manager Agent ────────────────────────┐
 client ──► │ 1 PLAN: agents, self-contained tasks, dependencies, success criteria (uses memory)   │
 (chat)     │ 2 DELEGATE: parallel waves; dependent tasks receive upstream findings               │
            │ 3 REVIEW: success criteria met? -> follow-up tasks                                 │
            │ 4 CONSOLIDATE -> 5 CRITIC review -> revise -> deliver -> remember turn             │
            └───────┬───────────────┬───────────────┬───────────────┬───────────────┬───────────┘
                    ▼               ▼               ▼               ▼               ▼
              Technology        Clinical           SEC         Corp. Resp.       Retail
         ReAct loop: thought -> tool call -> observation ... -> final_answer -> SELF-CHECK -> revise
```

| Agent | Tools it can choose from | Knowledge |
|---|---|---|
| Technology | `search_tech_docs`, `identify_router_model`, `diagnose_symptoms`, `rank_components`, `recommend_platforms` | Cisco datasheets, Google Cloud MCP/RAG |
| Clinical | `search_clinical_literature`, `recommend_agilent_products`, `search_clinical_trials`, `trial_landscape` | Agilent catalog, 20 clinical PDFs, 32k ClinicalTrials.gov studies |
| SEC | `list_filings`, `search_filings`, `risk_matrix`, `compare_companies` | Cisco, Agilent, Alphabet 10-Ks |
| Corporate Responsibility | `search_sofi`, `rank_investments` | FAO SOFI 2023 and 2024 |
| Retail | `identify_cart_items`, `recommend_addons`, `top_associations` | 1,000 transactions |

### What makes it agentic (not a pipeline)
- **Planning:** the Manager decides which agents to use and what each should do, with dependencies. Example: Technology recommends a platform, then SEC assesses that vendor's risk using Technology's findings.
- **Autonomous tool use:** agents choose tools and arguments, refine their searches when results are weak, and decide when they have enough evidence. The tool budget is configurable.
- **Self-correction:**
  - Every agent answer is checked for invalid citations, uncited claims, grounding against the cited passages, missing KPIs, and hypotheses stated as facts. If the check fails, the agent revises or searches again.
  - Malformed LLM output is retried, and unknown tool names are corrected.
- **Oversight:** the Manager reviews results against its success criteria and issues follow-up tasks. The Critic agent scores the draft (relevance, groundedness, actionability, risk awareness) and can force a revision. A revision is rejected if it makes the citations worse.
- **Memory:** follow-up questions such as "what about Agilent instead?" are resolved using earlier turns.
- **Graceful degradation:**
  - Rate limits trigger a circuit breaker.
  - Any agent that fails falls back to the deterministic pipeline.
  - With no LLM, the system runs fully rule-based and still produces a trace.
- **Transparency:** the UI shows the plan, a timeline of every thought, tool call, observation, self-check, review and critique, plus each agent's step-by-step reasoning.

## Team setup (clone and run)
The repository contains everything: the hackathon data, the code, downloaded web sources and the shared `.env`.
```bash
git clone <repo-url> hackathon
cd hackathon/consulting_platform
./run.sh            # first run: creates .venv + installs packages (~2 min), then builds indexes (~2 min)
```
The app opens at http://localhost:8501. Later runs start in seconds. Other commands: `./run.sh check`,
`./run.sh cli "question"`, `./run.sh eval`. Requires Python 3.10+ (`python3 --version`).
The team's shared Gemini key is already in `.env`, so keep this repository **private**. The free quota is shared by
everyone, so avoid running `./run.sh eval-llm` at the same time as a demo. If the data lives elsewhere, set
`DATA_ROOT=/path/to/hackathon` in `.env`.

## Quick start (macOS)
```bash
cd ~/Desktop/Project/256/hackathon/consulting_platform
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fetch_sources.py      # downloads the Cisco datasheets / Agilent & Google pages from the link files
python build_index.py        # ~1-2 min
python check_llm.py          # verifies the Gemini key in .env  -> "LLM OK"
streamlit run app.py         # http://localhost:8501
```
More ways to run it:
- `python cli.py "question"`: prints the full agent trace.
- `python cli.py --chat`: multi-turn chat in the terminal.
- `python cli.py --no-llm "question"`: rule-based mode, no LLM.
- `python evaluate.py`: offline metrics.
- `python evaluate.py --with-llm --agent-queries 5 --planner-limit 20`: adds agent-behaviour metrics. This uses many LLM calls, so watch your free-tier quota.

**Free-tier tips:** one request takes about 8–20 LLM calls (plan, about 3–4 per agent, review, consolidation, critic). If you hit rate limits, lower "Max tool calls per agent" or switch off review or critic in the sidebar. You can also set `LLM_MIN_INTERVAL=4` in `.env` to space calls out.

## Evaluation
`eval/results/report.md` covers:
1. Routing (keyword, prototype and hybrid routers, plus the LLM planner) on 52 test questions and 23 hold-out questions.
2. Retrieval: Hit@k and MRR on 37 questions with known answers.
3. Recommenders: market basket compared with popularity and random baselines, trial ranking P@5/nDCG, investment and fault-diagnosis checks.
4. Response quality: citation validity and grounding.
5. **Agent behaviour:**
   - share of runs completed by the ReAct loop
   - tool calls and distinct tools per agent, and tool error rate
   - self-revision rate and self-check pass rate
   - planner success rate and follow-up rate
   - critic approval and revision rates, and critic scores
   - LLM calls and latency per request

Offline results without an LLM: hybrid routing gets the main agent right 100% of the time and the exact set of agents 94.2% (hold-out: 91.3%). Retrieval reaches Hit@3 97.3% and MRR 0.92. Trial recommendations reach P@5 97.5%.

The market-basket recommender scores about the same as recommending the most popular items. The transactions contain little "bought together" signal, so the system reports lift with every suggestion and shows nothing when no reliable pattern exists.

## Project layout
```
core/agentic.py   ReAct agents, planning Manager, Critic, memory   ◄ agent core
core/tools.py     tool registry per agent (wraps retrieval + recommenders, evidence/citation registry)
core/quality.py   self-check / critic rules (citation validity, grounding, KPIs, hypothesis labelling)
core/trace.py     agent trace (UI timeline)
core/agents.py    domain logic + deterministic fallback agents
core/manager.py   routing-based orchestration + consolidation (fallback path)
core/router.py    intent detection (lexicon + prototype classifier); router hint for the planner
core/retriever.py BM25 hybrid retrieval;  core/ingest.py data ingestion;  core/basket.py market basket
core/llm.py       Gemini / OpenAI / Anthropic / Ollama REST client with throttling + circuit breaker
app.py  cli.py  api.py  evaluate.py  check_llm.py  fetch_sources.py  build_index.py
```
