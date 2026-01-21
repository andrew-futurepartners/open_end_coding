import json

from theme_governance import (
    GOVERNANCE_SCHEMA,
    THEME_SCHEMA,
    normalize_theme_dict_order,
    validate_json_schema,
)


def _valid_theme_dict():
    return {
        "major_themes": [
            {
                "id": "T2",
                "label": "Quality",
                "definition": "Comments about quality.",
                "approx_pct": 0.4,
                "subs": [
                    {
                        "id": "T2.1",
                        "label": "Durability",
                        "definition": "Mentions durability.",
                        "approx_pct": 0.2,
                        "examples": ["Lasts long", "Sturdy build"],
                    }
                ],
            },
            {
                "id": "T999",
                "label": "Non-answer",
                "definition": "Non-substantive responses.",
                "approx_pct": 0.1,
                "subs": [
                    {
                        "id": "T999.1",
                        "label": "Don't know",
                        "definition": "Does not know.",
                        "approx_pct": 0.1,
                        "examples": ["Don't know"],
                    }
                ],
            },
            {
                "id": "T1",
                "label": "Price",
                "definition": "Comments about price.",
                "approx_pct": 0.5,
                "subs": [
                    {
                        "id": "T1.1",
                        "label": "Too expensive",
                        "definition": "Mentions high cost.",
                        "approx_pct": 0.3,
                        "examples": ["Too pricey"],
                    }
                ],
            },
        ]
    }


def test_schema_validation_pass_fail():
    ok, err = validate_json_schema(_valid_theme_dict(), THEME_SCHEMA)
    assert ok, f"Expected schema to pass, got: {err}"

    invalid = {"major_themes": [{"id": "T1"}]}
    ok, err = validate_json_schema(invalid, THEME_SCHEMA)
    assert not ok, "Expected schema to fail for invalid theme dict"


def test_governance_schema_validation():
    governance_output = {
        "theme_dict": _valid_theme_dict(),
        "change_log": [
            {
                "action": "rename",
                "from_ids": ["T1.1"],
                "to_ids": ["T1.1"],
                "reason": "Clarify label",
            }
        ],
    }
    ok, err = validate_json_schema(governance_output, GOVERNANCE_SCHEMA)
    assert ok, f"Expected governance schema to pass, got: {err}"


def test_deterministic_ordering():
    theme_dict = _valid_theme_dict()
    ordered = normalize_theme_dict_order(theme_dict)
    labels = [m["label"] for m in ordered["major_themes"]]
    assert labels[-1] == "Non-answer", "Expected Non-answer to be ordered last"
    assert labels[:-1] == sorted(labels[:-1]), "Expected majors (non-answer) to be sorted"


if __name__ == "__main__":
    test_schema_validation_pass_fail()
    test_governance_schema_validation()
    test_deterministic_ordering()
    print(json.dumps({"status": "ok"}))
