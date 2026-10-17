"""Build all knowledge-base indexes.  Usage:  python build_index.py [technology clinical sec csr trials]"""
import sys
import time

import config
from core.retriever import build_all

if __name__ == "__main__":
    print(f"Data root: {config.DATA_ROOT}")
    t = time.time()
    stats = build_all(sys.argv[1:] or None)
    print(f"Done in {time.time()-t:.1f}s -> {config.INDEX_DIR}")
    if stats.get("technology", 0) < 30:
        print("\nNOTE: the Technology knowledge base is small. Run `python fetch_sources.py` to download the\n"
              "Cisco datasheets / Google Cloud pages listed in the hackathon link files, then rebuild.")
