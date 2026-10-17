"""Ingestion: turn the hackathon data folder into chunked, metadata-rich documents per domain.

Domains / knowledge bases
  technology : Cisco datasheets + product pages, Google Cloud MCP / RAG pages (fetched by fetch_sources.py)
  clinical   : clinical research PDFs, Agilent product pages / catalog, ClinicalTrials.gov studies (separate index)
  sec        : Form 10-K PDFs (Cisco, Agilent, Alphabet)
  csr        : FAO SOFI 2023 / 2024 reports
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import config
from core.text import chunk_text, clean_text

csv.field_size_limit(sys.maxsize)


# ----------------------------------------------------------------------------- PDF
def pdf_pages(path: Path) -> List[str]:
    """Return list of page texts (cached)."""
    cache_dir = config.INDEX_DIR / "pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{path}:{path.stat().st_size}".encode()).hexdigest()
    cache = cache_dir / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pages: List[str] = []
    try:
        import fitz  # PyMuPDF (fast)
        with fitz.open(path) as doc:
            pages = [p.get_text() for p in doc]
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
    cache.write_text(json.dumps(pages))
    return pages


def _pdf_docs(path: Path, base_meta: Dict) -> List[Dict]:
    docs = []
    for pno, text in enumerate(pdf_pages(path), start=1):
        for ci, ch in enumerate(chunk_text(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP)):
            d = dict(base_meta)
            d.update(page=pno, chunk=ci, text=ch)
            docs.append(d)
    return docs


# ----------------------------------------------------------------------------- link files
def parse_link_file(path: Path) -> List[Dict]:
    """Parse the instructor's link lists (title line followed by URL lines + optional description)."""
    entries, title = [], None
    if not path.exists():
        return entries
    for line in (l.strip() for l in path.read_text(errors="ignore").splitlines()):
        if not line or line in ("Cisco", "Google", "Agilent"):
            continue
        if line.startswith("http"):
            entries.append({"title": re.sub(r"^\d+\.\s*", "", title or path.stem), "url": line, "description": ""})
        elif re.match(r"^(Covers|\d+-page)", line) and entries:
            entries[-1]["description"] = line          # description of the preceding link
        else:
            title = line
    return entries


def _web_docs(domain: str) -> List[Dict]:
    """Pages downloaded by fetch_sources.py (data/web/<domain>/...)."""
    docs = []
    manifest_p = config.WEB_DIR / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    d = config.WEB_DIR / domain
    if not d.exists():
        return docs
    for f in sorted(d.iterdir()):
        meta = manifest.get(f.name, {})
        low = f.name.lower()
        guess = "Cisco" if any(k in low for k in ("cisco", "isr", "asr", "catalyst", "c8", "8000v", "8200", "8300",
                                                   "datasheet", "data_sheet", "c78")) else \
            "Agilent" if any(k in low for k in ("agilent", "seahorse", "nanozoomer", "pathology", "stain")) else \
            "Google" if any(k in low for k in ("google", "gcp", "mcp", "rag")) else ""
        title = meta.get("title") or f.stem.replace("_", " ").replace("-", " ")
        base = {"domain": domain, "source": title, "path": f"web/{domain}/{f.name}",
                "url": meta.get("url", ""), "doc_type": "web", "company": meta.get("company") or guess, "year": ""}
        if f.suffix == ".pdf":
            base["doc_type"] = "datasheet"
            docs += _pdf_docs(f, base)
        elif f.suffix == ".txt":
            text = f.read_text(errors="ignore")
            for ci, ch in enumerate(chunk_text(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP)):
                x = dict(base)
                x.update(page=1, chunk=ci, text=ch)
                docs.append(x)
    return docs


def _link_docs(domain: str) -> List[Dict]:
    docs = []
    for lf in config.LINK_FILES.get(domain, []):
        for e in parse_link_file(lf):
            company = "Cisco" if "cisco" in e["url"] else "Agilent" if "agilent" in e["url"] else \
                "Google" if "google" in e["url"] else ""
            docs.append({"domain": domain, "source": f"{e['title']} (catalog entry)", "path": lf.name,
                         "url": e["url"], "doc_type": "catalog", "company": company, "year": "", "page": 1,
                         "chunk": 0, "text": f"{e['title']}. {e['description']} Source: {e['url']}"})
    return docs


# ----------------------------------------------------------------------------- domain builders
def build_sec() -> List[Dict]:
    docs = []
    for pdf in sorted(config.SEC_DIR.glob("*/*.pdf")):
        company = pdf.parent.name
        first = " ".join(pdf_pages(pdf)[:2])
        m = re.search(r"fiscal year ended\s*:?\s*([A-Za-z]+\s+\d{1,2},\s*(\d{4}))", first, re.I)
        fy = m.group(2) if m else ""
        base = {"domain": "sec", "company": company, "year": fy, "doc_type": "10-K",
                "source": f"{company} Form 10-K (fiscal year ended {m.group(1) if m else 'n/a'})",
                "path": str(pdf.relative_to(config.DATA_ROOT)), "url": ""}
        docs += _pdf_docs(pdf, base)
    return docs


def build_csr() -> List[Dict]:
    docs = []
    for pdf in sorted(config.FOOD_DIR.glob("*.pdf")):
        m = re.search(r"(20\d\d)", pdf.stem)
        year = m.group(1) if m else ""
        base = {"domain": "csr", "company": "FAO", "year": year, "doc_type": "report",
                "source": f"FAO et al., The State of Food Security and Nutrition in the World {year}",
                "path": str(pdf.relative_to(config.DATA_ROOT)), "url": ""}
        docs += _pdf_docs(pdf, base)
    return docs


def build_clinical() -> List[Dict]:
    docs = []
    for pdf in sorted(config.CLINICAL_DIR.glob("*/*.pdf")):
        base = {"domain": "clinical", "company": "", "year": "", "doc_type": "research paper",
                "topic": pdf.parent.name, "source": pdf.stem.replace("_", " "),
                "path": str(pdf.relative_to(config.DATA_ROOT)), "url": ""}
        docs += _pdf_docs(pdf, base)
    docs += _link_docs("clinical") + _web_docs("clinical")
    return docs


def build_technology() -> List[Dict]:
    return _link_docs("technology") + _web_docs("technology")


def build_trials() -> List[Dict]:
    """ClinicalTrials.gov exports -> one record per study (used by the trial recommender)."""
    out = []
    for f in sorted(config.CLINICAL_DIR.glob("ctg-studies_*.csv")):
        topic = f.stem.replace("ctg-studies_", "")
        with open(f, newline="", encoding="utf-8", errors="ignore") as fh:
            for row in csv.DictReader(fh):
                summary = (row.get("Brief Summary") or "")[:700]
                text = (f"{row.get('Study Title','')}. Conditions: {row.get('Conditions','')}. "
                        f"Interventions: {row.get('Interventions','')}. Primary outcome: "
                        f"{(row.get('Primary Outcome Measures') or '')[:300]}. {summary}")
                out.append({
                    "domain": "clinical", "doc_type": "clinical trial", "topic": topic,
                    "nct": row.get("NCT Number", ""), "source": f"{row.get('NCT Number','')}: {row.get('Study Title','')}",
                    "url": row.get("Study URL", ""), "status": row.get("Study Status", ""),
                    "phase": row.get("Phases", ""), "sponsor": row.get("Sponsor", ""),
                    "enrollment": row.get("Enrollment", ""), "conditions": row.get("Conditions", ""),
                    "interventions": row.get("Interventions", ""), "study_type": row.get("Study Type", ""),
                    "results": row.get("Study Results", ""), "path": f.name, "page": 1, "chunk": 0,
                    "company": "", "year": (row.get("Start Date") or "")[:4], "text": clean_text(text),
                })
    return out


BUILDERS = {
    "technology": build_technology,
    "clinical": build_clinical,
    "sec": build_sec,
    "csr": build_csr,
    "trials": build_trials,
}
