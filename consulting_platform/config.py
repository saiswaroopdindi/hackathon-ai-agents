"""Central configuration for the Agentic AI Management Consulting Platform."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
# Hackathon data lives one level up (the folder that holds Agilent.txt, Clinical/, SEC/ ...)
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_DIR.parent)).resolve()
WEB_DIR = PROJECT_DIR / "data" / "web"          # pages downloaded by fetch_sources.py
INDEX_DIR = PROJECT_DIR / "data" / "index"      # built by build_index.py
EVAL_DIR = PROJECT_DIR / "eval"

SEC_DIR = DATA_ROOT / "UnitedStatesSecuritiesandExchangeCommission(SEC)"
FOOD_DIR = DATA_ROOT / "FoodSecurity"
CLINICAL_DIR = DATA_ROOT / "Clinical"
RETAIL_CSV = DATA_ROOT / "Retail" / "CMPE256_Hackathon_market_basket_analysis_Release.csv"
LINK_FILES = {
    "technology": [DATA_ROOT / "DataSource-Cisco.txt", DATA_ROOT / "Google GCP.txt"],
    "clinical": [DATA_ROOT / "Agilent.txt"],
}

# Chunking
CHUNK_CHARS = 1100
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 6

# Domain registry: used by router, agents and UI
DOMAINS = {
    "technology": {
        "label": "Technology Agent",
        "sources": "Cisco router datasheets, Google Cloud (MCP / RAG) docs",
        "color": "#2563eb",
    },
    "clinical": {
        "label": "Clinical Agent",
        "sources": "Agilent diagnostics catalog, clinical research PDFs, ClinicalTrials.gov studies",
        "color": "#059669",
    },
    "sec": {
        "label": "SEC Agent",
        "sources": "Cisco, Agilent, Alphabet Form 10-K filings",
        "color": "#7c3aed",
    },
    "csr": {
        "label": "Corporate Responsibility Agent",
        "sources": "FAO State of Food Security and Nutrition (SOFI) 2023 & 2024",
        "color": "#d97706",
    },
    "retail": {
        "label": "Retail / Procurement Recommender",
        "sources": "1,000 historical security-equipment transactions (market basket)",
        "color": "#db2777",
    },
}

# LLM provider: auto | gemini | openai | anthropic | ollama | none
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")  # auto-discovers if unavailable
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def load_dotenv() -> None:
    """Tiny .env loader (avoids a dependency)."""
    env = PROJECT_DIR / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()
# re-read after .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
