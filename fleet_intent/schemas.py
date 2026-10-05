from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_LENGTH = 500

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
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        examples=["disable my card 987654 forever"],
    )


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
    source: Literal["rules", "model"] = Field(
        ..., description="'rules' when the answer needed no model call"
    )
    latency_ms: float = Field(..., description="Time spent classifying, in milliseconds")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadinessResponse(BaseModel):
    ready: bool
    model: str
    ollama_reachable: bool
    model_available: bool


class ErrorResponse(BaseModel):
    detail: str
