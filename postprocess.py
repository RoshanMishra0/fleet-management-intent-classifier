"""Deterministic rules around the LLM.

The model decides what the user means. Fixed business rules (value mappings,
allowed actions, pagination phrases, unsupported request types) are enforced
here in code, where they are cheap, testable and always applied.

Two entry points:

* ``pre_classify(query)``  - requests that never need the model.
* ``postprocess(query, raw)`` - validate and normalise the model's output.
"""

import re
from typing import Optional

INTENTS = {
    "FUEL_SEARCH",
    "CARD_MANAGEMENT",
    "CARD_UNLOCK",
    "USER_MANAGEMENT",
    "FUEL_CODE",
    "PROMPTED_ID_MANAGEMENT",
    "NOTIFICATION_MANAGEMENT",
    "UNKNOWN",
}

ALLOWED_ACTIONS = {
    "FUEL_SEARCH": set(),
    "CARD_UNLOCK": set(),
    "UNKNOWN": set(),
    "CARD_MANAGEMENT": {"activate", "hold", "block"},
    "USER_MANAGEMENT": {"deactivate"},
    "FUEL_CODE": {"generate", "cancel", "list", "status"},
    "PROMPTED_ID_MANAGEMENT": {"create", "activate", "inactivate", "change", "details"},
    "NOTIFICATION_MANAGEMENT": {"enable", "disable", "enable_all", "disable_all", "details"},
}

# User wording -> business value, per intent. Order matters: the first verb
# found in the query wins, so stronger actions are listed first.
ACTION_SYNONYMS = {
    "CARD_MANAGEMENT": [
        ("block", ["delete", "remove", "cancel", "close", "lost", "stolen", "block"]),
        ("hold", ["deactivate", "disable", "suspend", "freeze", "pause", "hold"]),
        ("activate", ["reactivate", "activate"]),
    ],
    "FUEL_CODE": [
        ("cancel", ["cancel", "void"]),
        ("status", ["status", "check"]),
        ("list", ["list", "show"]),
        ("generate", ["generate", "create", "get"]),
    ],
    "PROMPTED_ID_MANAGEMENT": [
        ("inactivate", ["deactivate", "disable", "inactivate"]),
        ("change", ["update", "modify", "change"]),
        ("details", ["show", "get", "list", "details"]),
        ("create", ["create"]),
        ("activate", ["activate"]),
    ],
    "NOTIFICATION_MANAGEMENT": [
        ("details", ["show", "list", "details"]),
        ("disable", ["disable", "deactivate"]),
        ("enable", ["enable", "activate"]),
    ],
}

DEFAULT_ACTION = {"FUEL_CODE": "generate", "USER_MANAGEMENT": "deactivate"}

PERMANENCE_WORDS = ["permanently", "permanent", "forever", "for good"]

PAGINATION_PHRASES = ["next option", "another one", "show me another", "what else"]

BUSINESS_OBJECTS = [
    "fuel", "truck stop", "truck station", "gas station", "gas stop", "card",
    "user", "email", "fuel code", "pin", "pid", "prompted id", "driver id",
    "notification", "notifications", "alert", "alerts", "exception", "exceptions",
]

ANALYTICS_WORDS = [
    "analytics", "report", "reports", "trend", "trends", "insight", "insights",
    "dashboard", "dashboards", "summary", "summaries", "summarize", "summarise",
]

FUEL_TYPES = [
    ("Reefer", ["reefer"]),
    ("DEF", ["def"]),
    ("CNG", ["cng"]),
    ("LNG", ["lng"]),
    ("Diesel", ["diesel"]),
    ("Gas", ["gasoline", "petrol", "gas"]),
]

MERCHANT_TYPES = {"TS", "GS", "FS"}
FUEL_TYPE_VALUES = {value for value, _ in FUEL_TYPES}
AMENITY_CODES = {"ATM", "TrkPrkPave", "TrkPrkUnPv", "TrkPrkOvNt"}

SLOT_FIELDS_BY_INTENT = {
    "FUEL_SEARCH": {"merchant_type", "fuel_type", "radius", "radius_unit", "amenities"},
    "CARD_MANAGEMENT": {"card_number"},
    "CARD_UNLOCK": {"card_number"},
    "USER_MANAGEMENT": {"email"},
    "FUEL_CODE": set(),
    "PROMPTED_ID_MANAGEMENT": set(),
    "NOTIFICATION_MANAGEMENT": set(),
    "UNKNOWN": set(),
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CARD_RE = re.compile(r"(?<!\d)\d{4,6}(?!\d)")
RADIUS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(miles?|mi|kilometers?|kilometres?|kms?)\b")


def _has(text: str, phrase: str) -> bool:
    """Whole-word (or whole-phrase) match, so 'pin' does not match 'shopping'."""
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None


def _has_any(text: str, phrases) -> bool:
    return any(_has(text, p) for p in phrases)


def _empty_result(intent: str = "UNKNOWN") -> dict:
    return {
        "intent": intent,
        "action": None,
        "merchant_type": None,
        "fuel_type": None,
        "radius": None,
        "radius_unit": None,
        "amenities": [],
        "card_number": None,
        "email": None,
        "pagination_action": None,
        "confidence": 0.0,
    }


def is_analytics_request(q: str) -> bool:
    """Reports, dashboards and summaries are out of scope -> UNKNOWN.

    'report card 1234 stolen' is a card request, not a reporting request.
    """
    if not _has_any(q, ANALYTICS_WORDS):
        return False
    if _has(q, "card") and _has_any(q, ["lost", "stolen"]):
        return False
    return True


def has_pagination(q: str) -> bool:
    return any(p in q for p in PAGINATION_PHRASES)


def pre_classify(query: str) -> Optional[dict]:
    """Return a final answer for requests that do not need the model."""
    q = query.lower().strip()

    if is_analytics_request(q):
        result = _empty_result("UNKNOWN")
        result["confidence"] = 1.0
        return result

    if has_pagination(q) and not _has_any(q, BUSINESS_OBJECTS):
        result = _empty_result("UNKNOWN")
        result["pagination_action"] = "next"
        result["confidence"] = 1.0
        return result

    return None


def normalize_action(intent: str, raw_action, q: str):
    allowed = ALLOWED_ACTIONS.get(intent, set())
    if not allowed:
        return None

    if intent == "USER_MANAGEMENT":
        return "deactivate"

    if intent == "CARD_MANAGEMENT" and _has_any(q, PERMANENCE_WORDS):
        return "block"

    if intent == "NOTIFICATION_MANAGEMENT":
        if _has(q, "enable all"):
            return "enable_all"
        if _has(q, "disable all"):
            return "disable_all"

    raw = raw_action.strip().lower() if isinstance(raw_action, str) else None

    # 1. The model already returned a valid business value.
    if raw in allowed:
        return raw

    synonyms = ACTION_SYNONYMS.get(intent, [])

    # 2. The model returned the user's verb: map it.
    if raw:
        for value, verbs in synonyms:
            if raw in verbs:
                return value

    # 3. The model returned nothing usable: read the verb from the query.
    for value, verbs in synonyms:
        if _has_any(q, verbs):
            return value

    return DEFAULT_ACTION.get(intent)


def extract_amenities(q: str) -> list:
    amenities = []
    if _has(q, "parking"):
        specific = False
        if _has(q, "unpaved parking"):
            amenities.append("TrkPrkUnPv")
            specific = True
        if _has(q, "paved parking"):
            amenities.append("TrkPrkPave")
            specific = True
        if _has(q, "overnight parking"):
            amenities.append("TrkPrkOvNt")
            specific = True
        if not specific:
            amenities.extend(["TrkPrkPave", "TrkPrkUnPv", "TrkPrkOvNt"])
    if _has(q, "atm"):
        amenities.append("ATM")
    return amenities


def extract_merchant_type(q: str):
    truck = _has_any(q, ["truck stop", "truck stops", "truck station", "truck stations"])
    gas = _has_any(q, ["gas station", "gas stations", "gas stop", "gas stops", "gasoline station", "gasoline stations"])
    if truck and gas:
        return "FS"
    if truck:
        return "TS"
    if gas:
        return "GS"
    return None


def extract_fuel_type(q: str, merchant_type):
    # 'gas station' names a place, not a fuel, so look at the query without it.
    stripped = re.sub(r"gas(oline)? (station|stop)s?", " ", q)
    for value, words in FUEL_TYPES:
        if _has_any(stripped, words):
            return value
    if merchant_type == "GS":
        return "Gas"
    return "Diesel"


def extract_radius(q: str):
    match = RADIUS_RE.search(q)
    if not match:
        return None, None
    number = float(match.group(1))
    radius = int(number) if number.is_integer() else number
    unit = "miles" if match.group(2).startswith("mi") else "km"
    return radius, unit


def normalize_unit(unit):
    if not isinstance(unit, str):
        return None
    unit = unit.strip().lower()
    if unit in {"miles", "mile", "mi"}:
        return "miles"
    if unit in {"km", "kms", "kilometers", "kilometres", "kilometer", "kilometre"}:
        return "km"
    return None


def postprocess(query: str, raw: dict) -> dict:
    """Validate the model output and apply the fixed business rules."""
    q = query.lower().strip()
    raw = raw if isinstance(raw, dict) else {}

    intent = raw.get("intent")
    if not (isinstance(intent, str) and intent in INTENTS):
        intent = "UNKNOWN"

    result = _empty_result(intent)

    confidence = raw.get("confidence")
    if isinstance(confidence, (int, float)):
        result["confidence"] = max(0.0, min(1.0, float(confidence)))

    if has_pagination(q):
        result["pagination_action"] = "next"

    result["action"] = normalize_action(intent, raw.get("action"), q)

    fields = SLOT_FIELDS_BY_INTENT[intent]

    # Slots: keep the model's value when it is valid, fall back to a rule
    # when it is missing or outside the allowed vocabulary.
    if "amenities" in fields:
        merchant = raw.get("merchant_type")
        if not (isinstance(merchant, str) and merchant in MERCHANT_TYPES):
            merchant = extract_merchant_type(q)
        result["merchant_type"] = merchant

        fuel = raw.get("fuel_type")
        if not (isinstance(fuel, str) and fuel in FUEL_TYPE_VALUES):
            fuel = extract_fuel_type(q, merchant)
        result["fuel_type"] = fuel

        radius, unit = raw.get("radius"), normalize_unit(raw.get("radius_unit"))
        if not isinstance(radius, (int, float)) or isinstance(radius, bool) or unit is None:
            radius, unit = extract_radius(q)
        result["radius"], result["radius_unit"] = radius, unit

        # Amenity codes are a closed list and "parking" has a fixed expansion,
        # so merge the model's valid codes with the rule-derived ones.
        model_codes = raw.get("amenities") if isinstance(raw.get("amenities"), list) else []
        merged = []
        for code in model_codes:
            if isinstance(code, str) and code in AMENITY_CODES and code not in merged:
                merged.append(code)
        for code in extract_amenities(q):
            if code not in merged:
                merged.append(code)
        result["amenities"] = merged

    if "card_number" in fields:
        card = raw.get("card_number")
        card = str(card) if card is not None else None
        if not (card and CARD_RE.fullmatch(card) and card in q):
            match = CARD_RE.search(q)
            card = match.group(0) if match else None
        result["card_number"] = card

    if "email" in fields:
        email = raw.get("email")
        if not (isinstance(email, str) and EMAIL_RE.fullmatch(email) and email.lower() in q):
            match = EMAIL_RE.search(query)
            email = match.group(0) if match else None
        result["email"] = email

    return result
