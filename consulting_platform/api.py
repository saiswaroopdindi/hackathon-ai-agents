"""Optional FastAPI microservice.   Run:  uvicorn api:app --reload --port 8000

POST /basket/recommend  {"cart": ["Bosch D7050 Detector (SKU: D7050)"], "k": 5}
POST /consult           {"query": "...", "strategy": "auto", "domains": null}
GET  /basket/items
"""
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from core.basket import BasketRecommender, exact_cart_seen
from core.agentic import AgenticManager

app = FastAPI(title="Agentic AI Consulting Platform")
_basket = BasketRecommender()
_mgr = None


class CartReq(BaseModel):
    cart: List[str]
    k: int = 5


class ConsultReq(BaseModel):
    query: str
    strategy: str = "auto"
    domains: Optional[List[str]] = None


@app.get("/basket/items")
def items():
    return _basket.items


@app.post("/basket/recommend")
def recommend(req: CartReq):
    recs = _basket.recommend(req.cart, req.k)
    return {"cart": req.cart, "exact_combination_seen": exact_cart_seen(_basket.tx, req.cart),
            "recommendations": [r.to_dict() for r in recs],
            "message": None if recs else "No reliable pattern for this cart; no recommendation shown."}


@app.post("/consult")
def consult(req: ConsultReq):
    global _mgr
    _mgr = _mgr or AgenticManager()
    return _mgr.ask(req.query, strategy=req.strategy, force_domains=req.domains).to_dict()
