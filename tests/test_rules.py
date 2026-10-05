"""Unit tests for the rule layer. No model or server needed: `pytest`."""

import json
from pathlib import Path

from fleet_intent.rules import (
    extract_amenities,
    extract_radius,
    is_analytics_request,
    normalize_action,
    postprocess,
    pre_classify,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def same(a, b):
    return sorted(a) == sorted(b) if isinstance(a, list) and isinstance(b, list) else a == b


# --- pre_classify: requests that never reach the model ----------------------

def test_analytics_requests_are_unknown():
    for q in ["generate fuel usage report", "summarize monthly fuel activity",
              "show fuel spend trends dashboard", "show analytics insights for drivers"]:
        result = pre_classify(q)
        assert result is not None and result["intent"] == "UNKNOWN"
        assert result["action"] is None and result["pagination_action"] is None


def test_reporting_a_stolen_card_is_not_analytics():
    assert not is_analytics_request("report card 8642 stolen")
    assert pre_classify("report card 8642 stolen") is None


def test_standalone_pagination():
    for q in ["show me another", "what else", "next option", "another one"]:
        result = pre_classify(q)
        assert result["intent"] == "UNKNOWN"
        assert result["pagination_action"] == "next"


def test_pagination_with_business_object_goes_to_model():
    assert pre_classify("show me another truck stop") is None
    result = postprocess("show me another truck stop", {"intent": "FUEL_SEARCH"})
    assert result["pagination_action"] == "next"
    assert result["merchant_type"] == "TS"


def test_ordinary_requests_go_to_model():
    for q in ["activate card 1234", "find nearest truck stop", "enable notification"]:
        assert pre_classify(q) is None


def test_word_boundaries():
    # "pin" inside "shopping" is not a business object.
    assert pre_classify("what else is there for shopping")["intent"] == "UNKNOWN"


# --- normalize_action --------------------------------------------------------

def test_card_actions_map_to_business_values():
    assert normalize_action("CARD_MANAGEMENT", "remove", "remove card 5555") == "block"
    assert normalize_action("CARD_MANAGEMENT", "disable", "disable card 5555") == "hold"
    assert normalize_action("CARD_MANAGEMENT", "reactivate", "reactivate card 5555") == "activate"
    assert normalize_action("CARD_MANAGEMENT", None, "my card 778899 was stolen") == "block"


def test_deactivate_is_not_read_as_activate():
    assert normalize_action("CARD_MANAGEMENT", None, "deactivate card 2468") == "hold"


def test_permanence_overrides_model():
    assert normalize_action("CARD_MANAGEMENT", "hold", "disable my card 987654 forever") == "block"
    assert normalize_action("CARD_MANAGEMENT", "hold", "freeze card 7788 for good") == "block"


def test_valid_model_action_is_kept():
    assert normalize_action("FUEL_CODE", "status", "check fuel code status") == "status"
    assert normalize_action("PROMPTED_ID_MANAGEMENT", "details", "show prompted id details") == "details"


def test_user_management_is_always_deactivate():
    assert normalize_action("USER_MANAGEMENT", None, "remove user john@test.com") == "deactivate"
    assert normalize_action("USER_MANAGEMENT", "remove", "remove user john@test.com") == "deactivate"


def test_fuel_code_synonyms_and_default():
    assert normalize_action("FUEL_CODE", "create", "create a fuel code") == "generate"
    assert normalize_action("FUEL_CODE", "void", "void fuel code") == "cancel"
    assert normalize_action("FUEL_CODE", None, "I need a fuel code") == "generate"


def test_prompted_id_synonyms():
    assert normalize_action("PROMPTED_ID_MANAGEMENT", "deactivate", "disable pid") == "inactivate"
    assert normalize_action("PROMPTED_ID_MANAGEMENT", "modify", "modify driver id") == "change"


def test_notification_all_variants():
    assert normalize_action("NOTIFICATION_MANAGEMENT", "disable", "disable all alerts") == "disable_all"
    assert normalize_action("NOTIFICATION_MANAGEMENT", "enable", "enable all notifications") == "enable_all"
    assert normalize_action("NOTIFICATION_MANAGEMENT", "activate", "activate alerts") == "enable"


def test_intents_without_actions():
    for intent in ["FUEL_SEARCH", "CARD_UNLOCK", "UNKNOWN"]:
        assert normalize_action(intent, "show", "anything") is None


# --- slot rules --------------------------------------------------------------

def test_generic_parking_expands_to_three_codes():
    assert sorted(extract_amenities("truck stop with parking and atm")) == sorted(
        ["TrkPrkPave", "TrkPrkUnPv", "TrkPrkOvNt", "ATM"])


def test_specific_parking_is_not_expanded():
    assert extract_amenities("gas station with unpaved parking") == ["TrkPrkUnPv"]
    assert extract_amenities("gas stop with paved parking") == ["TrkPrkPave"]
    assert extract_amenities("station with overnight parking") == ["TrkPrkOvNt"]


def test_radius_units():
    assert extract_radius("within 15 mi") == (15, "miles")
    assert extract_radius("within 30 km") == (30, "km")
    assert extract_radius("nearest truck stop") == (None, None)


# --- postprocess -------------------------------------------------------------

def test_invalid_intent_becomes_unknown():
    assert postprocess("hello", {"intent": "SOMETHING_ELSE"})["intent"] == "UNKNOWN"
    assert postprocess("hello", "not a dict")["intent"] == "UNKNOWN"


def test_model_values_outside_vocabulary_are_replaced():
    result = postprocess("find nearest truck stop within 20 miles",
                         {"intent": "FUEL_SEARCH", "merchant_type": ["TS"], "fuel_type": "petroleum",
                          "radius": "20", "radius_unit": "mi", "amenities": ["WiFi"]})
    assert result["merchant_type"] == "TS"
    assert result["fuel_type"] == "Diesel"
    assert (result["radius"], result["radius_unit"]) == (20, "miles")
    assert result["amenities"] == []


def test_hallucinated_card_number_is_replaced():
    result = postprocess("activate card 1234", {"intent": "CARD_MANAGEMENT", "action": "activate", "card_number": "9999"})
    assert result["card_number"] == "1234"


def test_irrelevant_slots_are_cleared():
    result = postprocess("activate card 1234",
                         {"intent": "CARD_MANAGEMENT", "action": "activate", "fuel_type": "Diesel", "email": "a@b.com"})
    assert result["fuel_type"] is None and result["email"] is None


def test_confidence_is_clamped():
    assert postprocess("activate card 1234", {"intent": "CARD_MANAGEMENT", "confidence": 7})["confidence"] == 1.0


# --- the labelled test sets agree with the rules ----------------------------

def _rules_only(case):
    """What the service returns if the model gets only the intent right."""
    ruled = pre_classify(case["query"])
    if ruled is not None:
        return ruled
    return postprocess(case["query"], {"intent": case["expected"]["intent"]})


def _check_file(name):
    with open(DATA / name, encoding="utf-8") as f:
        cases = json.load(f)
    for case in cases:
        prediction = _rules_only(case)
        for key, value in case["expected"].items():
            assert same(prediction.get(key), value), (case["query"], key, prediction.get(key), value)


def test_original_cases_are_consistent_with_rules():
    _check_file("test_cases.json")


def test_heldout_cases_are_consistent_with_rules():
    _check_file("test_cases_heldout.json")
