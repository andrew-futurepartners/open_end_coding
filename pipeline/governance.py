from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple, Callable

from pipeline.llm_client import (
    CacheStats,
    JsonParseError,
    RateLimiter,
    SQLiteCache,
    oai_json_completion,
    retry_with_backoff,
    repair_json_to_schema,
    validate_or_repair_json,
)
from theme_governance import (
    GOVERNANCE_SCHEMA,
    THEME_SCHEMA,
    apply_governance_change_log,
    normalize_change_log,
    normalize_theme_dict_order,
    validate_json_schema,
)


GOVERNANCE_SYSTEM = (
    "You are a senior qualitative research methodologist. "
    "You reduce overlap, normalize naming, and enforce consistent granularity in a theme dictionary. "
    "You preserve meaning and avoid inventing new concepts."
)

GOVERNANCE_USER_TEMPLATE = (
    """
You will review and improve a hierarchical theme dictionary for a survey open-end question.

Goals:
- Reduce duplicate or overlapping sub-themes.
- Normalize naming for clarity and consistency.
- Normalize granularity so peers are at similar specificity.
- Preserve meaning and avoid new concepts unless strictly necessary.
- DO NOT change the Non-answer major theme or its sub-themes.
- Keep existing IDs whenever possible. Only create new IDs if a split is absolutely required.
- If you must add IDs, use deterministic numbering within the parent major theme (e.g., T2.7).

Return JSON that includes:
1) "theme_dict": the full corrected dictionary (schema enforced).
2) "change_log": a list of actions describing the changes.

Allowed change_log actions:
- merge: merge several sub-themes into one
- split: split one sub-theme into several (use sparingly)
- rename: rename a theme without changing meaning
- move: move a sub-theme to a different major theme

Each change_log entry must include:
{
  "action": "merge|split|rename|move",
  "from_ids": ["..."],
  "to_ids": ["..."],
  "reason": "short explanation"
}

Question context (if available):
{question_context}

Theme dictionary:
{theme_json}
"""
)


def govern_theme_dict(
    client,
    model: str,
    theme_dict: Dict[str, Any],
    question_context: Dict[str, Any] | None,
    seed: int,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    on_notice: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
    if not theme_dict or not theme_dict.get("major_themes"):
        return theme_dict, [], {"prompt_tokens": 0, "completion_tokens": 0}

    # #region agent log
    try:
        majors_in = theme_dict.get("major_themes", []) if isinstance(theme_dict, dict) else []
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H13",
                "location": "pipeline/governance.py:govern_theme_dict:input",
                "message": "Governance input themes",
                "data": {
                    "major_count": len(majors_in),
                    "major_labels": [m.get("label", "") for m in majors_in][:8],
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion

    theme_json = json.dumps(theme_dict, ensure_ascii=False, separators=(",", ":"))
    qc_text = json.dumps(question_context or {}, ensure_ascii=False, separators=(",", ":"))
    try:
        user = GOVERNANCE_USER_TEMPLATE.replace("{question_context}", qc_text).replace("{theme_json}", theme_json)
    except Exception:
        raise

    def make_request():
        return oai_json_completion(
            client,
            model,
            GOVERNANCE_SYSTEM,
            user,
            seed,
            response_schema=GOVERNANCE_SCHEMA,
            limiter=limiter,
            cache=cache,
            cache_stats=cache_stats,
            reasoning_effort="medium",
            verbosity="low",
            reserve_output_tokens=8_000,
        )

    try:
        data, usage, raw, _ = retry_with_backoff(make_request)
    except JsonParseError as e:
        data, usage, raw = repair_json_to_schema(
            client,
            model,
            e.raw_text,
            GOVERNANCE_SCHEMA,
            seed,
            limiter,
            cache,
            cache_stats,
            "Theme governance",
        )

    data, repair_usage, repaired = validate_or_repair_json(
        client,
        model,
        data,
        raw,
        GOVERNANCE_SCHEMA,
        seed,
        limiter,
        cache,
        cache_stats,
        "Theme governance",
    )
    if repaired and on_notice:
        on_notice("Theme governance output required JSON repair; validation succeeded after repair.")
        usage["prompt_tokens"] += repair_usage.get("prompt_tokens", 0)
        usage["completion_tokens"] += repair_usage.get("completion_tokens", 0)

    proposed_theme_dict = data.get("theme_dict", {})
    change_log = normalize_change_log(data.get("change_log", []))

    governed = apply_governance_change_log(theme_dict, proposed_theme_dict, change_log)
    governed = normalize_theme_dict_order(governed)

    # #region agent log
    try:
        majors_out = governed.get("major_themes", []) if isinstance(governed, dict) else []
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H14",
                "location": "pipeline/governance.py:govern_theme_dict:output",
                "message": "Governance output themes",
                "data": {
                    "major_count": len(majors_out),
                    "major_labels": [m.get("label", "") for m in majors_out][:8],
                    "change_log_count": len(change_log or []),
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion

    ok, err = validate_json_schema(governed, THEME_SCHEMA)
    if not ok:
        raise ValueError(f"Theme governance produced invalid theme_dict: {err}")

    return governed, change_log, usage
