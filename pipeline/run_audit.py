from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dataset_signature(texts: List[str], question_text: str) -> Dict[str, Any]:
    unique_texts = sorted({t or "" for t in texts})
    payload = json.dumps(
        {
            "row_count": len(texts),
            "unique_count": len(unique_texts),
            "question_text": question_text,
            "unique_texts": unique_texts,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "row_count": len(texts),
        "unique_count": len(unique_texts),
        "question_hash": _hash_text(question_text or ""),
        "data_hash": _hash_text(payload),
    }


def compute_run_id(
    dataset_sig: Dict[str, Any],
    model: str,
    seed: int,
    prompt_versions: Dict[str, str],
    settings: Dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "dataset": dataset_sig,
            "model": model,
            "seed": seed,
            "prompts": prompt_versions,
            "settings": settings,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _hash_text(payload)


def confidence_histogram(assignments: List[Dict[str, Any]]) -> Dict[str, int]:
    bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for item in assignments:
        conf = item.get("confidence")
        if not isinstance(conf, (int, float)):
            conf = 0.0
        if conf < 0.2:
            bins["0.0-0.2"] += 1
        elif conf < 0.4:
            bins["0.2-0.4"] += 1
        elif conf < 0.6:
            bins["0.4-0.6"] += 1
        elif conf < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def build_run_audit(
    run_id: str,
    dataset_sig: Dict[str, Any],
    question_text: str,
    theme_params: Dict[str, Any],
    governance_log: List[Dict[str, Any]],
    assignment_params: Dict[str, Any],
    assignments: List[Dict[str, Any]],
    cache_stats: Dict[str, Any],
    usage_totals: Dict[str, Any],
) -> Dict[str, Any]:
    needs_review = sum(1 for item in assignments if item.get("decision") == "needs_review")
    return {
        "run_id": run_id,
        "input_signature": {
            "row_count": dataset_sig.get("row_count"),
            "unique_count": dataset_sig.get("unique_count"),
            "question_hash": dataset_sig.get("question_hash"),
            "question_text": question_text,
        },
        "theme_generation": {
            "params": theme_params,
            "governance_log": governance_log,
        },
        "assignment": {
            "params": assignment_params,
            "needs_review_count": needs_review,
            "confidence_histogram": confidence_histogram(assignments),
        },
        "usage_totals": usage_totals,
        "cache_stats": cache_stats,
    }
