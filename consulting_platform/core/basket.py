"""Market-basket recommender (association rules) for checkout integration.

Given the current cart, find rules  {antecedent subset of cart} -> item  learned from historical
transactions, score by confidence/lift, prefer rules whose antecedent covers more of the cart,
and return nothing (gracefully) when no reliable pattern exists.
"""
from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence

import config


@dataclass
class BasketRec:
    item: str
    score: float
    confidence: float
    lift: float
    support_count: int
    antecedent: List[str]

    def to_dict(self):
        return {"item": self.item, "score": round(self.score, 4), "confidence": round(self.confidence, 3),
                "lift": round(self.lift, 2), "support_count": self.support_count, "because_of": self.antecedent}


def load_transactions(path=None) -> List[List[str]]:
    path = path or config.RETAIL_CSV
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            items = [v.strip() for k, v in row.items() if k.startswith("item") and v and v.strip()]
            if items:
                out.append(sorted(set(items)))
    return out


class BasketRecommender:
    def __init__(self, transactions: Optional[List[List[str]]] = None, max_len: int = 3,
                 min_count: int = 3, min_lift: float = 1.1):
        self.tx = transactions if transactions is not None else load_transactions()
        self.n = len(self.tx)
        self.max_len, self.min_count, self.min_lift = max_len, min_count, min_lift
        self.counts: Counter = Counter()
        for t in self.tx:
            for k in range(1, max_len + 1):
                for c in combinations(t, k):
                    self.counts[frozenset(c)] += 1
        self.items = sorted({i for t in self.tx for i in t})
        self.pop = Counter({i: self.counts[frozenset([i])] for i in self.items})

    def support(self, itemset) -> int:
        return self.counts.get(frozenset(itemset), 0)

    def recommend(self, cart: Sequence[str], k: int = 5) -> List[BasketRec]:
        cart = [c for c in dict.fromkeys(cart) if c in self.pop]
        if not cart:
            return []
        best: Dict[str, BasketRec] = {}
        for size in range(min(len(cart), self.max_len - 1), 0, -1):
            for ante in combinations(sorted(cart), size):
                a_cnt = self.support(ante)
                if a_cnt < self.min_count:
                    continue
                for item in self.items:
                    if item in cart:
                        continue
                    both = self.support(set(ante) | {item})
                    if both < self.min_count:
                        continue
                    conf = both / a_cnt
                    lift = conf / (self.pop[item] / self.n)
                    if lift < self.min_lift:
                        continue
                    # larger antecedents describe the cart better -> bonus; lift rewards true association
                    score = conf * (1 + 0.35 * (size - 1)) * min(lift, 3.0) ** 0.5
                    if item not in best or score > best[item].score:
                        best[item] = BasketRec(item, score, conf, lift, both, list(ante))
        return sorted(best.values(), key=lambda r: -r.score)[:k]

    def popular(self, k: int = 5, exclude: Sequence[str] = ()) -> List[str]:
        return [i for i, _ in self.pop.most_common() if i not in exclude][:k]

    def top_rules(self, k: int = 10) -> List[Dict]:
        rules = []
        for s, cnt in self.counts.items():
            if len(s) != 2 or cnt < self.min_count:
                continue
            a, b = sorted(s)
            for x, y in ((a, b), (b, a)):
                conf = cnt / self.pop[x]
                lift = conf / (self.pop[y] / self.n)
                rules.append({"if": x, "then": y, "support": round(cnt / self.n, 3), "confidence": round(conf, 3),
                              "lift": round(lift, 2), "count": cnt})
        return sorted(rules, key=lambda r: (-r["lift"], -r["count"]))[:k]


def evaluate(k: int = 5, seeds=(1, 2, 3, 4, 5), test_frac: float = 0.2) -> Dict:
    """Average of several random 80/20 splits (the dataset is small, single splits are noisy)."""
    runs = [_evaluate_once(k, s, test_frac) for s in seeds]
    out = {key: (round(sum(r[key] for r in runs) / len(runs), 3) if isinstance(runs[0][key], float)
                 else runs[0][key]) for key in runs[0]}
    out["splits"] = len(runs)
    out["hit_rate_std"] = round(float(__import__("statistics").pstdev([r[f"hit_rate@{k}"] for r in runs])), 3)
    return out


def exact_cart_seen(tx: List[List[str]], cart: Sequence[str]) -> int:
    s = set(cart)
    return sum(1 for t in tx if s <= set(t))


def _evaluate_once(k: int = 5, seed: int = 7, test_frac: float = 0.2) -> Dict:
    """Leave-one-out evaluation on held-out baskets vs a popularity baseline."""
    tx = load_transactions()
    rnd = random.Random(seed)
    rnd.shuffle(tx)
    cut = int(len(tx) * (1 - test_frac))
    train, test = tx[:cut], tx[cut:]
    model = BasketRecommender(train)
    hits = pop_hits = rr = cases = answered = 0
    prec_sum = 0.0
    for t in test:
        if len(t) < 2:
            continue
        for held in t:
            cart = [i for i in t if i != held]
            recs = [r.item for r in model.recommend(cart, k)]
            pops = model.popular(k, exclude=cart)
            cases += 1
            if recs:
                answered += 1
                rel = set(t) - set(cart)
                prec_sum += len(rel & set(recs)) / len(recs)
            if held in recs:
                hits += 1
                rr += 1 / (recs.index(held) + 1)
            if held in pops:
                pop_hits += 1
    n_items = len(model.items)
    return {"test_baskets": len(test), "cases": cases, f"hit_rate@{k}": round(hits / cases, 3),
            f"popularity_hit_rate@{k}": round(pop_hits / cases, 3),
            f"random_expected_hit_rate@{k}": round(k / max(n_items - 2, 1), 3),
            "mrr": round(rr / cases, 3), "coverage(answered)": round(answered / cases, 3),
            f"precision@{k}(answered)": round(prec_sum / max(answered, 1), 3),
            "lift_over_popularity": round((hits / cases) / max(pop_hits / cases, 1e-9), 2)}
