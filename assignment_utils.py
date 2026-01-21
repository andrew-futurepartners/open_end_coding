from __future__ import annotations

from typing import Any, Dict, List, Tuple


def make_assignment_decision_schema(allowed_ids: List[str], max_codes: int) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "assignment_decisions",
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "idx": {"type": "integer"},
                                "subtheme_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": int(max_codes),
                                    "items": {"type": "string", "enum": allowed_ids},
                                },
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "decision": {"type": "string", "enum": ["ok", "needs_review"]},
                                "rationale": {"type": "string"},
                            },
                            "required": ["idx", "subtheme_ids", "confidence", "decision", "rationale"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def normalize_candidate_ids(candidate_ids: List[str]) -> List[str]:
    return sorted({str(x) for x in (candidate_ids or []) if x})


def stable_candidate_key(candidate_ids: List[str]) -> Tuple[str, ...]:
    return tuple(normalize_candidate_ids(candidate_ids))


def coerce_decision(confidence: float, low_thresh: float, decision: str | None) -> str:
    if confidence <= 0.4:
        return "needs_review"
    if confidence < low_thresh:
        return "needs_review"
    if decision == "ok":
        return "ok"
    return "needs_review"


def build_assignments_list(subtheme_ids: List[str], confidence: float) -> List[Dict[str, Any]]:
    return [{"theme_id": sid, "confidence": float(confidence)} for sid in subtheme_ids]


def normalize_assignment_result(
    result: Dict[str, Any],
    candidate_ids: List[str],
    all_subtheme_ids: List[str],
    max_codes: int,
    low_thresh: float,
) -> Dict[str, Any]:
    candidates = normalize_candidate_ids(candidate_ids)
    all_ids = set(all_subtheme_ids or [])
    candidate_set = set(candidates)
    errors: List[str] = []

    subtheme_ids = result.get("subtheme_ids")
    if not isinstance(subtheme_ids, list):
        subtheme_ids = []
        errors.append("missing subtheme_ids")

    cleaned: List[str] = []
    for sid in subtheme_ids:
        if sid not in all_ids:
            errors.append(f"invalid id: {sid}")
            continue
        if candidate_set and sid not in candidate_set:
            errors.append(f"id not in candidates: {sid}")
            continue
        cleaned.append(sid)

    if not cleaned:
        fallback = candidates[0] if candidates else (all_subtheme_ids[0] if all_subtheme_ids else "")
        if fallback:
            cleaned = [fallback]
            errors.append("fallback applied")

    cleaned = cleaned[: max_codes]

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
        errors.append("invalid confidence")
    confidence = float(max(0.0, min(1.0, confidence)))

    decision = coerce_decision(confidence, low_thresh, result.get("decision"))

    rationale = result.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
        errors.append("invalid rationale")

    if errors:
        note = "; ".join(errors[:2])
        rationale = (rationale + " " + f"[Validation: {note}]").strip()
        decision = "needs_review"

    return {
        "subtheme_ids": cleaned,
        "confidence": confidence,
        "decision": decision,
        "rationale": rationale,
    }
