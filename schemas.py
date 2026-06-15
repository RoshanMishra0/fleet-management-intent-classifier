from typing import List, Optional, Literal
from pydantic import BaseModel, Field


IntentType = Literal[
    "FUEL_SEARCH",
    "CARD_MANAGEMENT",
    "CARD_UNLOCK",
    "USER_MANAGEMENT",
    "FUEL_CODE",
    "PROMPTED_ID_MANAGEMENT",
    "NOTIFICATION_MANAGEMENT",
    "UNKNOWN",
]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class ClassificationResponse(BaseModel):
    intent: IntentType
    action: Optional[str] = None
    merchant_type: Optional[str] = None
    fuel_type: Optional[str] = None
    radius: Optional[float] = None
    radius_unit: Optional[str] = None
    amenities: List[str] = []
    card_number: Optional[str] = None
    email: Optional[str] = None
    pagination_action: Optional[str] = None
    confidence: Optional[float] = None