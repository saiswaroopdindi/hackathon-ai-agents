"""Quick LLM connectivity test.   python check_llm.py"""
from core.llm import get_llm

llm = get_llm()
print("Provider:", llm.label)
if llm.available:
    try:
        print("Reply:", llm.generate("Reply with exactly: LLM OK", max_tokens=20).strip())
        print("Model used:", llm.model)
    except Exception as e:
        print("FAILED:", e)
