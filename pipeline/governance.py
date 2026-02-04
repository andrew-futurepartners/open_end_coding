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

_ATTRACTION_KEYWORDS = {
    "park",
    "canyon",
    "resort",
    "memorial",
    "monument",
    "museum",
    "trail",
    "bridge",
    "island",
    "beach",
    "lake",
    "mount",
    "mountain",
    "falls",
    "valley",
    "forest",
    "zoo",
    "aquarium",
    "stadium",
    "arena",
    "statue",
    "tower",
    "national park",
    "historic",
    "landmark",
}


def _contains_attraction_keyword(label: str) -> bool:
    text = (label or "").lower()
    return any(k in text for k in _ATTRACTION_KEYWORDS)


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
    # Restore dropped attraction/landmark labels (guard against over-merge).
    try:
        input_map: Dict[str, Dict[str, Any]] = {}
        input_major_for: Dict[str, str] = {}
        for m in (theme_dict.get("major_themes", []) if isinstance(theme_dict, dict) else []):
            mid = m.get("id")
            for s in (m.get("subs", []) or []):
                lbl = (s.get("label") or "").strip()
                if lbl:
                    input_map[lbl.lower()] = s
                    if mid:
                        input_major_for[lbl.lower()] = mid
        output_labels = set()
        output_by_id: Dict[str, Dict[str, Any]] = {}
        for m in (governed.get("major_themes", []) if isinstance(governed, dict) else []):
            for s in (m.get("subs", []) or []):
                lbl = (s.get("label") or "").strip()
                if lbl:
                    output_labels.add(lbl.lower())
                sid = str(s.get("id", "") or "")
                if sid and sid not in output_by_id:
                    output_by_id[sid] = s
        restored = []
        relabeled = []
        for lbl_key, sub in input_map.items():
            label_text = sub.get("label", "") or ""
            is_general = "(general)" in label_text.lower()
            if lbl_key not in output_labels and (_contains_attraction_keyword(label_text) or is_general):
                sid = str(sub.get("id", "") or "")
                existing = output_by_id.get(sid) if sid else None
                if existing and is_general:
                    existing_label = (existing.get("label") or "")
                    if "(general)" not in existing_label.lower():
                        existing["label"] = label_text
                        relabeled.append(f"{existing_label} -> {label_text}")
                    continue
                mid = input_major_for.get(lbl_key)
                if not mid:
                    continue
                target_major = next((m for m in governed.get("major_themes", []) or [] if m.get("id") == mid), None)
                if not target_major:
                    continue
                target_major.setdefault("subs", []).append(sub)
                restored.append(sub.get("label", ""))
        if restored:
            # De-duplicate by ID after restoring to avoid duplicate labels for same subtheme.
            try:
                for m in (governed.get("major_themes", []) if isinstance(governed, dict) else []):
                    seen_ids = set()
                    deduped = []
                    for s in (m.get("subs", []) or []):
                        sid = str(s.get("id", "") or "")
                        if sid and sid in seen_ids:
                            continue
                        if sid:
                            seen_ids.add(sid)
                        deduped.append(s)
                    m["subs"] = deduped
            except Exception:
                pass
            governed = normalize_theme_dict_order(governed)
    except Exception:
        pass

    ok, err = validate_json_schema(governed, THEME_SCHEMA)
    if not ok:
        raise ValueError(f"Theme governance produced invalid theme_dict: {err}")

    return governed, change_log, usage
