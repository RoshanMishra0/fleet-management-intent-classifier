# Fleet Management Intent Classifier POC

## Overview

This project is a local LLM-based intent classification and slot extraction system for fleet management queries.

The system takes a natural language user query and returns a structured JSON response containing the predicted intent and extracted slots such as action, merchant type, fuel type, radius, amenities, card number, email, and pagination action.

## Supported Intents

* FUEL_SEARCH
* CARD_MANAGEMENT
* CARD_UNLOCK
* USER_MANAGEMENT
* FUEL_CODE
* PROMPTED_ID_MANAGEMENT
* NOTIFICATION_MANAGEMENT
* UNKNOWN

## Tech Stack

* Python
* FastAPI
* Ollama
* qwen2.5:3b local model
* Pydantic

## Setup

Install dependencies:

pip install -r requirements.txt

Pull the local model using Ollama:

ollama pull qwen2.5:3b

Run the FastAPI server:

uvicorn app:app --reload

Open API docs:

http://127.0.0.1:8000/docs

## API Usage

Endpoint:

POST /classify

Sample request:

{
  "query": "disable my card 987654 forever"
}

Sample response:

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

## Evaluation Metrics

The system was evaluated on 45 manually prepared test cases covering all supported intents and key business rules.

The local LLM achieved an intent accuracy of 91.11%, slot extraction accuracy of 89.68%, and exact match accuracy of 77.78%. The system returned valid JSON for all test cases, giving a valid JSON rate of 100%. The average response latency during evaluation was 4.487 seconds per query.

Exact match accuracy is lower than intent and slot accuracy because it requires every expected field in a test case to match perfectly. Even a single slot mismatch marks the full case as incorrect.

## Running Evaluation

Start the FastAPI server first:

uvicorn app:app --reload

Then run:

python evaluate.py

The evaluation output is saved in:
metrics_report.json