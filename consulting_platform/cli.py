"""Command-line demo of the agentic platform.

  python cli.py "Our Cisco ISR4331 router drops connections intermittently ..."     # uses the LLM in .env
  python cli.py --no-llm "..."                                                       # rule-based agents
  python cli.py --chat                                                               # multi-turn conversation
"""
import argparse
import json

from core.agentic import AgenticManager
from core.llm import LLM, get_llm


def show(r, verbose=True):
    print(f"\nLLM: {r.llm} | {len(r.agents)} agent(s) | {sum(a.tool_calls for a in r.agents)} tool calls | "
          f"{r.latency_s}s | LLM calls {r.llm_calls}")
    print("\n=== AGENT TRACE ===")
    for e in r.trace:
        print(f"{e['t']:6.1f}s  {e['actor'][:30]:30s} {e['kind']:12s} {str(e['title'])[:150]}")
    if verbose:
        for a in r.agents:
            print(f"\n--- {a.label} [{a.confidence.get('level')} {a.confidence.get('score')}] mode={a.mode}")
            print(a.answer)
    print("\n=== CONSULTING RECOMMENDATION ===\n" + r.recommendation)
    if r.critique:
        print(f"\nCritic: {r.critique.get('verdict')} {r.critique.get('scores', '')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--steps", type=int, default=4, help="max tool calls per agent")
    ap.add_argument("--domains", nargs="*", help="force agents, e.g. technology sec")
    a = ap.parse_args()
    mgr = AgenticManager(LLM("none") if a.no_llm else get_llm(), max_steps=a.steps)
    if a.chat:
        print("Multi-turn consulting chat (empty line to quit)")
        while True:
            q = input("\nclient> ").strip()
            if not q:
                break
            show(mgr.ask(q, force_domains=a.domains), verbose=False)
    elif a.query:
        r = mgr.ask(a.query, force_domains=a.domains)
        if a.json:
            print(json.dumps(r.to_dict(), indent=2, default=str))
        else:
            show(r)
    else:
        ap.print_help()
