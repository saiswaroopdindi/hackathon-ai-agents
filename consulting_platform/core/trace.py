"""Thread-safe trace of everything the agents do (shown in the UI as a timeline)."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

ICONS = {"plan": "🧭", "delegate": "📨", "thought": "💭", "tool": "🔧", "observation": "👁", "reflection": "🔍",
         "revise": "✏️", "review": "📋", "followup": "↩️", "consolidate": "🧩", "critique": "⚖️", "final": "✅",
         "memory": "🧠", "fallback": "⚠️", "error": "❌"}


class Trace:
    def __init__(self, on_event: Optional[Callable[[Dict], None]] = None):
        self.events: List[Dict] = []
        self._lock = threading.Lock()
        self._t0 = time.time()
        self.on_event = on_event

    def log(self, actor: str, kind: str, title: str, detail: Any = None) -> Dict:
        ev = {"t": round(time.time() - self._t0, 2), "actor": actor, "kind": kind, "icon": ICONS.get(kind, "•"),
              "title": title, "detail": detail}
        with self._lock:
            self.events.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass
        return ev

    def by_actor(self) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        for e in self.events:
            out.setdefault(e["actor"], []).append(e)
        return out

    def to_list(self) -> List[Dict]:
        return list(self.events)
