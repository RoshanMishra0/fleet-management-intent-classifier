Fleet Management Intent Classifier

Intent classification and slot extraction for fleet-management requests, running entirely on a local 3B-parameter LLM. A free-text query goes in; validated JSON that an application can act on comes out. No query, card number or email leaves the machine.

"disable my card 987654 forever"   →   { "intent": "CARD_MANAGEMENT", "action": "block", "card_number": "987654", ... }
Results

Evaluated on 45 hand-written test cases covering all eight intents and the main business rules.

Metric	Result
Intent accuracy	91.11% (41 of 45)
Slot accuracy	89.68%
Exact match (every field correct)	77.78% (35 of 45)
Valid JSON	100%
Average latency	4.49 s per query

Model: qwen2.5:3b on Ollama, temperature 0, JSON mode. Measured on: [add your machine here: CPU or GPU, RAM]

Exact match is the strict number: one wrong slot fails the whole case. The full per-case output is in metrics_report.json.

How it works
User query
FastAPI /classify
Prompt: intent rules, slotmappings, examples
qwen2.5:3b on Ollama,JSON mode
Pydantic validation
Structured response
app.py exposes POST /classify and validates the request with Pydantic.
llm_extractor.py builds one prompt containing the intent definitions, the slot normalisation rules (for example "truck stop" → TS, "disable" → hold, "forever" → block), conflict-resolution rules and two worked examples.
Ollama runs the model in JSON mode at temperature 0, so the same query gives the same answer.
The output is parsed and validated against ClassificationResponse in schemas.py. If the model output cannot be parsed, the service returns UNKNOWN with confidence 0 instead of failing.
Why a small local model

The task needs instruction following, classification, extraction, value normalisation and JSON output in a single call. A 3B generative model handles all five, and it runs on local hardware: there is no per-request API cost, and queries containing card numbers and email addresses stay local.

Supported intents
Intent	Covers	Normalised actions
FUEL_SEARCH	Find, nearest or cheapest fuel locations. Slots: merchant type (TS, GS, FS), fuel type, radius and unit, amenities	none
CARD_MANAGEMENT	Activate, pause or cancel a card	activate, hold, block
CARD_UNLOCK	Only when the word "unlock" appears	none
USER_MANAGEMENT	Remove or suspend a user; any query with an email address	deactivate
FUEL_CODE	Fuel code requests	generate, cancel, list, status
PROMPTED_ID_MANAGEMENT	PIN, PID, prompted ID or driver ID requests	create, activate, inactivate, change, details
NOTIFICATION_MANAGEMENT	Notifications, alerts and exceptions	enable, disable, enable_all, disable_all, details
UNKNOWN	Analytics and report requests, standalone pagination, anything unsupported	none

A few of the business rules the model has to apply:

Fuel type defaults to Diesel when a fuel search does not name one.
"Parking" with no qualifier expands to all three parking amenity codes.
"Permanently", "forever" or "for good" turns any card action into block.
"Next option", "another one", "show me another" and "what else" set pagination_action to next.
Error analysis

Ten of the 45 cases failed exact match. They fall into four groups.

Failure	Cases	Example
Action missing or not mapped to the business value	5	"remove card 5555" returned remove; expected block
Unsupported request classified as NOTIFICATION_MANAGEMENT instead of UNKNOWN	3	"generate fuel usage report"
Pagination phrase missed	1	"what else" returned no pagination_action
Amenity list incomplete	1	generic "parking" returned one code instead of three

The pattern matters more than the count. In nine of the ten failures the model understood the request but did not apply a fixed rule. Fixed rules are cheaper and more reliable in code than in a prompt, which is what the first two roadmap items address.

Known limitations
Small test set. With 45 cases, one case moves a metric by about 2.2 points.
Not fully held out. One test query is also one of the two worked examples in the prompt.
Latency. About 4.5 seconds is too slow for an interactive assistant. The full rule prompt is sent on every request.
Loose schema. Only intent is constrained to an allowed set; action and the slot fields are free strings.
Confidence is not calibrated. The confidence field is the model's own estimate.
Valid JSON depends on JSON mode. The 100% rate comes from Ollama's JSON mode, with the UNKNOWN fallback behind it.

Roadmap
 Post-processing layer: per-intent action enums and a synonym map (targets 5 of the 10 failures)
 Rule pre-check for standalone pagination and analytics keywords (targets 4 of the 10)
 Held-out test set of 200+ cases with paraphrases and typos, excluding the prompt examples
 Side-by-side comparison with two other small local models
 Lower latency: shorter prompt, system prompt kept warm through the chat endpoint, schema-constrained output
 Unit tests and pinned dependencies
 
Quick start
Requires Python 3 and Ollama.

bash
pip install -r requirements.txt
ollama pull qwen2.5:3b
uvicorn app:app --reload

Interactive API docs: http://127.0.0.1:8000/docs

Request
bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "disable my card 987654 forever"}'
Response
json
{
  "intent": "CARD_MANAGEMENT",
  "action": "block",
  "merchant_type": null,
  "fuel_type": null,
  "radius": null,
  "radius_unit": null,
  "amenities": [],
  "card_number": "987654",
  "email": null,
  "pagination_action": null,
  "confidence": 0.95
}
Run the evaluation

With the server running:

bash
python evaluate.py

The script sends every case in test_cases.json to /classify, compares each field with the expected value, prints the failures side by side, and writes the metrics and per-case results to metrics_report.json.

Project structure
File	Purpose
app.py	FastAPI service with the /classify endpoint
llm_extractor.py	Prompt, Ollama call and JSON fallback
schemas.py	Pydantic request and response models
evaluate.py	Evaluation runner and metrics
test_cases.json	Test queries with expected output
metrics_report.json	Latest evaluation results
Tech stack

Python, FastAPI, Pydantic, Ollama, Qwen 2.5 (3B)
