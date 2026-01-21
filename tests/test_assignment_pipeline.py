from assignment_utils import (
    make_assignment_decision_schema,
    normalize_assignment_result,
    stable_candidate_key,
)
from theme_governance import validate_json_schema


def test_assignment_schema_conformity():
    schema = make_assignment_decision_schema(["T1.1", "T1.2"], max_codes=2)
    payload = {
        "results": [
            {
                "idx": 0,
                "subtheme_ids": ["T1.1"],
                "confidence": 0.8,
                "decision": "ok",
                "rationale": "Matches definition."
            }
        ]
    }
    ok, err = validate_json_schema(payload, schema)
    assert ok, f"Schema validation failed: {err}"


def test_id_validity_and_review_flag():
    result = {
        "idx": 0,
        "subtheme_ids": ["BAD"],
        "confidence": 0.9,
        "decision": "ok",
        "rationale": "Test"
    }
    normalized = normalize_assignment_result(
        result,
        candidate_ids=["T1.1"],
        all_subtheme_ids=["T1.1", "T1.2"],
        max_codes=2,
        low_thresh=0.6,
    )
    assert normalized["decision"] == "needs_review"
    assert normalized["subtheme_ids"] == ["T1.1"]


def test_deterministic_candidate_key():
    key1 = stable_candidate_key(["T1.2", "T1.1"])
    key2 = stable_candidate_key(["T1.1", "T1.2"])
    assert key1 == key2


if __name__ == "__main__":
    test_assignment_schema_conformity()
    test_id_validity_and_review_flag()
    test_deterministic_candidate_key()
    print({"status": "ok"})
