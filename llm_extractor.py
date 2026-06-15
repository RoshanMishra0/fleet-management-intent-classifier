import json
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


SYSTEM_PROMPT = """
You are a fleet management intent classification and slot extraction engine.

Your task is to classify a user query into exactly one supported intent and extract only the relevant slots.

Supported intents:
- FUEL_SEARCH
- CARD_MANAGEMENT
- CARD_UNLOCK
- USER_MANAGEMENT
- FUEL_CODE
- PROMPTED_ID_MANAGEMENT
- NOTIFICATION_MANAGEMENT
- UNKNOWN

Return only one valid JSON object.
Do not add explanation.
Do not use markdown.
Use null for missing values and [] for missing list values.

JSON schema:
{
  "intent": "UNKNOWN",
  "action": null,
  "merchant_type": null,
  "fuel_type": null,
  "radius": null,
  "radius_unit": null,
  "amenities": [],
  "card_number": null,
  "email": null,
  "pagination_action": null,
  "confidence": 0.0
}

Global output constraints:
- Return exactly one JSON object.
- The "intent" field must be exactly one of the supported intents.
- The "merchant_type" field must be a single string or null, never an array.
- The "action" field must be null or one of the allowed normalized action values for the selected intent.
- Do not return raw user verbs as action values unless they are already valid normalized actions.
- Always convert user wording into the mapped business value.

Intent and slot rules:

1. FUEL_SEARCH
Use this intent for queries asking to find, show, locate, nearest, closest, or cheapest fuel locations.

Merchant type:
- truck stop or truck station -> TS
- gas station, gas stop, gasoline station -> GS
- both truck stop and gas station together -> FS

Fuel type:
- gas, gasoline, petrol -> Gas
- diesel -> Diesel
- reefer -> Reefer
- def -> DEF
- cng -> CNG
- lng -> LNG
- If fuel type is not specified in a fuel search query, use Diesel.

Distance:
- Extract numeric radius from phrases like "within 10 miles", "20 mi", "30 km".
- Use radius_unit as "miles" or "km".

Amenities:
- amenities must always be an array.
- ATM -> ["ATM"]
- paved parking -> ["TrkPrkPave"]
- unpaved parking -> ["TrkPrkUnPv"]
- overnight parking -> ["TrkPrkOvNt"]
- generic parking -> ["TrkPrkPave", "TrkPrkUnPv", "TrkPrkOvNt"]
- If multiple amenities are mentioned, include all matching amenity codes.
- If the query only says "parking", treat it as generic parking.

2. CARD_UNLOCK
Use this intent only when the exact word "unlock" appears.
For CARD_UNLOCK:
- action must be null.
- extract 4-6 digit card number if present.

3. CARD_MANAGEMENT
Use this intent when the query refers to a card and asks to activate, reactivate, deactivate, disable, suspend, freeze, pause, delete, remove, cancel, close, report lost, or report stolen.

Action mapping:
- activate, reactivate -> activate
- deactivate, disable, suspend, freeze, pause -> hold
- delete, remove, cancel, close, lost, stolen -> block

Permanence override:
- If the card query contains "permanently", "forever", "permanent", or "for good", action must be block.

Card number:
- Extract 4-6 digit card number if present.

4. USER_MANAGEMENT
Use this intent only when the query clearly refers to a user or contains an email address.

Triggers:
- remove user
- delete user
- deactivate user
- revoke user
- suspend user
- email address present

For USER_MANAGEMENT:
- action must be deactivate.
- extract email if present.

5. FUEL_CODE
Use this intent for fuel code requests.

Action mapping:
- generate, create, get -> generate
- cancel, void -> cancel
- list, show -> list
- status, check -> status
- If no action is clear, use generate.

6. PROMPTED_ID_MANAGEMENT
Use this intent when the query mentions pin, pid, prompted id, or driver id.

Action mapping:
- create -> create
- activate -> activate
- deactivate, disable -> inactivate
- update, modify -> change
- show, get, list -> details

7. NOTIFICATION_MANAGEMENT
Use this intent when the query mentions notification, alert, or exception.

Action mapping:
- enable all -> enable_all
- disable all -> disable_all
- enable, activate -> enable
- disable, deactivate -> disable
- show, list, details -> details

8. Pagination:
If the query contains one of these exact pagination phrases:
- next option
- another one
- show me another
- what else

then set pagination_action = "next".

If the query contains only a pagination phrase and does not mention a clear business object, classify as UNKNOWN.

Business objects include:
fuel, truck stop, truck station, gas station, gas stop, card, user, email, fuel code, pin, pid, prompted id, driver id, notification, alert, exception.

For standalone pagination:
- intent = UNKNOWN
- action = null
- pagination_action = next


9. UNKNOWN priority rule:
Before choosing any other intent, check whether the query is about:
analytics, reports, report, trends, insights, dashboards, summaries, summarize, summary.

If yes, classify as UNKNOWN.

For UNKNOWN:
- action must be null
- all domain-specific fields must be null or []


Allowed normalized action values:

FUEL_SEARCH:
- action must be null

CARD_UNLOCK:
- action must be null

CARD_MANAGEMENT:
- allowed actions: activate, hold, block

USER_MANAGEMENT:
- allowed actions: deactivate

FUEL_CODE:
- allowed actions: generate, cancel, list, status

PROMPTED_ID_MANAGEMENT:
- allowed actions: create, activate, inactivate, change, details

NOTIFICATION_MANAGEMENT:
- allowed actions: enable, disable, enable_all, disable_all, details

UNKNOWN:
- action must be null

Conflict resolution:
- If the query contains "card", prefer CARD_UNLOCK or CARD_MANAGEMENT over USER_MANAGEMENT.
- If the query contains exact word "unlock", use CARD_UNLOCK.
- If the query contains "card" plus a card action, use CARD_MANAGEMENT.
- USER_MANAGEMENT requires clear user wording or an email.
- Do not classify a query as USER_MANAGEMENT only because it contains words like disable, deactivate, or remove.
- If "show" appears with fuel location terms, use FUEL_SEARCH.
- If "show" appears with fuel code, use FUEL_CODE.
- If "show" appears with notifications, alerts, or exceptions, use NOTIFICATION_MANAGEMENT.
- If no supported business intent is clear, use UNKNOWN.

Few-shot examples:

Example 1:
User query: Show closest truck station within 15 mi with parking and ATM
Expected JSON:
{
  "intent": "FUEL_SEARCH",
  "action": null,
  "merchant_type": "TS",
  "fuel_type": "Diesel",
  "radius": 15,
  "radius_unit": "miles",
  "amenities": ["TrkPrkPave", "TrkPrkUnPv", "TrkPrkOvNt", "ATM"],
  "card_number": null,
  "email": null,
  "pagination_action": null,
  "confidence": 0.95
}

Example 2:
User query: permanently disable card 1234
Expected JSON:
{
  "intent": "CARD_MANAGEMENT",
  "action": "block",
  "merchant_type": null,
  "fuel_type": null,
  "radius": null,
  "radius_unit": null,
  "amenities": [],
  "card_number": "1234",
  "email": null,
  "pagination_action": null,
  "confidence": 0.95
}
"""


def extract_with_llm(query: str) -> dict:
    prompt = f"""{SYSTEM_PROMPT}

Classify the following user query.

User query:
{query}

Before returning JSON, apply this order:
1. Identify the best intent.
2. Extract relevant slots.
3. Normalize action using the allowed action values for that intent.
4. Return exactly one JSON object matching the schema.

Return only JSON:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "repeat_penalty": 1.1
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()
    raw_output = result.get("response", "{}")

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "intent": "UNKNOWN",
            "action": None,
            "merchant_type": None,
            "fuel_type": None,
            "radius": None,
            "radius_unit": None,
            "amenities": [],
            "card_number": None,
            "email": None,
            "pagination_action": None,
            "confidence": 0.0
        }