from typing import Literal
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    type: Literal["person", "org", "product", "location", "other"]


class TicketExtraction(BaseModel):
    intent: Literal["bug_report", "feature_request", "billing", "other"]
    urgency: Literal["low", "medium", "high"]
    entities: list[Entity] = Field(default_factory=list)
    summary: str = Field(..., max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)


class InvoiceExtraction(BaseModel):
    vendor: str
    invoice_number: str
    amount: float
    currency: Literal["USD", "EUR", "GBP", "BDT", "other"]
    due_date: str  # ISO date, or "not specified"
    confidence: float = Field(..., ge=0.0, le=1.0)


SCHEMA_REGISTRY = {
    "ticket": TicketExtraction,
    "invoice": InvoiceExtraction,
}
