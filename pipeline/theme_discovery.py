from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Callable

from pipeline.llm_client import (
    CacheStats,
    JsonParseError,
    RateLimiter,
    SQLiteCache,
    oai_json_completion,
    retry_with_backoff,
    validate_or_repair_json,
    repair_json_to_schema,
)
from theme_governance import THEME_SCHEMA
from pipeline.utils import chunk_data, estimate_tokens, safe_prompt_token_budget


THEME_DISCOVERY_SYSTEM = (
    "You are a senior market research analyst. You design clear, business-ready thematic taxonomies "
    "that provide actionable insights. Focus on the substantive content of what respondents are saying "
    "about the topic, not on survey mechanics or response quality. Use neutral, professional language."
)

THEME_DISCOVERY_USER = (
    """
You will read a set of open-ended responses for one survey question.
Create a hierarchical coding frame with Major Themes and Sub-themes.
Goals:
- Capture the full variety by creating specific, meaningful Sub-themes.
- Keep Major Themes distinct and non-overlapping.
- Each Sub-theme should ladder under exactly one Major Theme.
- Use neutral, professional language.
- Assume responses are already translated to English.

Return JSON only with this schema:
{
  "major_themes": [
    {
      "id": "T1",
      "label": "<Major label>",
      "definition": "<one sentence>",
      "approx_pct": 0.00,
      "subs": [
        {
          "id": "T1.1",
          "label": "<Sub label>",
          "definition": "<one sentence>",
          "approx_pct": 0.00,
          "examples": ["ex1", "ex2", "ex3"]
        }
      ]
    }
  ]
}
Return only valid JSON matching the schema; do not include any additional text.
"""
)


def merge_theme_chunks(theme_chunks: List[Dict[str, Any]], on_notice: Callable[[str], None] | None = None) -> List[Dict[str, Any]]:
    if not theme_chunks:
        return []

    def norm_label(x: str) -> str:
        return re.sub(r"\s+", " ", (x or "").strip().lower())

    def norm_alpha(x: str) -> str:
        return re.sub(r"[^a-z]+", "", (x or "").lower())

    all_major_labels = set()
    all_sub_labels = set()
    for theme in theme_chunks:
        major_label = theme.get("label", "")
        if major_label:
            all_major_labels.add(norm_label(major_label))
        for sub in theme.get("subs", []) or []:
            sub_label = sub.get("label", "")
            if sub_label:
                all_sub_labels.add(norm_label(sub_label))

    conflicting_labels = {lbl for lbl in (all_major_labels & all_sub_labels) if lbl}
    if conflicting_labels and on_notice:
        on_notice(f"Resolved {len(conflicting_labels)} hierarchy conflicts.")

    merged_by_label: Dict[str, Dict[str, Any]] = {}
    for theme in theme_chunks:
        major_label = (theme.get("label") or "").strip()
        if not major_label:
            continue

        major_key = norm_label(major_label)
        merged_major = merged_by_label.get(major_key)
        if not merged_major:
            merged_major = {
                "id": theme.get("id", ""),
                "label": major_label,
                "definition": theme.get("definition", ""),
                "approx_pct": theme.get("approx_pct", 0.0),
                "subs": [],
            }
            merged_by_label[major_key] = merged_major

        existing_sub_keys = {norm_label(s.get("label", "")): s for s in merged_major.get("subs", []) if s.get("label")}
        for sub in theme.get("subs", []) or []:
            sub_label = (sub.get("label") or "").strip()
            if not sub_label:
                continue
            sub_key = norm_label(sub_label)
            if sub_key in conflicting_labels or sub_key in existing_sub_keys:
                continue
            merged_major["subs"].append({
                "id": sub.get("id", ""),
                "label": sub_label,
                "definition": sub.get("definition", ""),
                "approx_pct": sub.get("approx_pct", 0.0),
                "examples": sub.get("examples", []),
            })
            existing_sub_keys[sub_key] = sub

    merged_themes = list(merged_by_label.values())
    for theme in merged_themes:
        major_label_norm = norm_label(theme.get("label", ""))
        theme["subs"] = [sub for sub in theme.get("subs", []) if norm_label(sub.get("label", "")) != major_label_norm]

    nonanswer = []
    normal = []
    for theme in merged_themes:
        if norm_alpha(theme.get("label", "")) == "nonanswer":
            nonanswer.append(theme)
        else:
            normal.append(theme)

    normal.sort(key=lambda t: norm_label(t.get("label", "")))
    nonanswer.sort(key=lambda t: norm_label(t.get("label", "")))
    ordered = normal + nonanswer

    next_major_num = 1
    for theme in ordered:
        if norm_alpha(theme.get("label", "")) == "nonanswer":
            major_id = "T999"
        else:
            major_id = f"T{next_major_num}"
            next_major_num += 1
        theme["id"] = major_id
        subs = theme.get("subs", []) or []
        subs.sort(key=lambda s: norm_label(s.get("label", "")))
        for j, sub in enumerate(subs, start=1):
            sub["id"] = f"{major_id}.{j}"
        theme["subs"] = subs

    return ordered


def build_theme_frame_with_progress(
    client,
    model: str,
    texts: List[str],
    freq: List[int],
    seed: int,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    question_context: Dict[str, Any] | None = None,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
    on_notice: Callable[[str], None] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    def progress(val: int) -> None:
        if on_progress:
            on_progress(val)

    status("Preparing theme discovery...")
    progress(10)
    filtered_data = [
        {"text": t, "weight": int(w)}
        for t, w in zip(texts, freq)
        if str(t or "").strip() != ""
    ]
    filtered_data.sort(key=lambda x: x["weight"], reverse=True)

    safe_limit = safe_prompt_token_budget(model, reserve_output_tokens=12_000)
    total_tokens = estimate_tokens(
        json.dumps(filtered_data, ensure_ascii=False, separators=(",", ":")),
        model=model,
    )
    force_chunking = True

    enhanced_prompt = THEME_DISCOVERY_USER
    if question_context and question_context.get("type") != "general":
        enhanced_prompt += f"\n\nQUESTION CONTEXT: {question_context.get('focus', '')}\n"
        if question_context.get("priority_themes"):
            enhanced_prompt += f"PRIORITY THEMES: {', '.join(question_context['priority_themes'])}\n"
        enhanced_prompt += "Consider these priorities when creating your thematic framework.\n"

    if (not force_chunking) and (total_tokens <= safe_limit):
        payload = json.dumps(filtered_data, ensure_ascii=False, separators=(",", ":"))
        status("Generating themes...")
        progress(40)
        user = enhanced_prompt + "\n\nWeighted responses (JSON array):\n" + payload

        def make_request():
            return oai_json_completion(
                client,
                model,
                THEME_DISCOVERY_SYSTEM,
                user,
                seed,
                response_schema=THEME_SCHEMA,
                limiter=limiter,
                cache=cache,
                cache_stats=cache_stats,
                reasoning_effort="medium",
                verbosity="low",
                reserve_output_tokens=12_000,
            )

        single_ok = True
        try:
            data, usage, raw, _ = retry_with_backoff(
                make_request,
                on_retry=(lambda msg: on_notice("Retrying theme chunk after timeout...") if on_notice and "timeout" in msg else None),
            )
        except JsonParseError as e:
            data, usage, raw = repair_json_to_schema(
                client,
                model,
                e.raw_text,
                THEME_SCHEMA,
                seed,
                limiter,
                cache,
                cache_stats,
                "Theme discovery",
            )
        except Exception as e:
            msg = str(e).lower()
            if "timeout" in msg:
                if on_notice:
                    on_notice("Theme discovery timed out; retrying with smaller chunks.")
                single_ok = False
            else:
                raise

        if single_ok:
            data, repair_usage, repaired = validate_or_repair_json(
                client,
                model,
                data,
                raw,
                THEME_SCHEMA,
                seed,
                limiter,
                cache,
                cache_stats,
                "Theme discovery",
            )
            if repaired:
                usage["prompt_tokens"] += repair_usage.get("prompt_tokens", 0)
                usage["completion_tokens"] += repair_usage.get("completion_tokens", 0)

            progress(100)
            return data, usage

    status("Chunking theme discovery...")
    chunk_budget = int(safe_limit * 0.80)
    chunks = chunk_data(filtered_data, max_tokens=chunk_budget, model=model)
    all_themes = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for i, chunk in enumerate(chunks, start=1):
        payload = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        user = enhanced_prompt + "\n\nWeighted responses (JSON array):\n" + payload

        def make_request():
            return oai_json_completion(
                client,
                model,
                THEME_DISCOVERY_SYSTEM,
                user,
                seed,
                response_schema=THEME_SCHEMA,
                limiter=limiter,
                cache=cache,
                cache_stats=cache_stats,
                reasoning_effort="medium",
                verbosity="low",
                reserve_output_tokens=12_000,
            )

        try:
            data, usage, raw, _ = retry_with_backoff(make_request)
        except JsonParseError as e:
            data, usage, raw = repair_json_to_schema(
                client,
                model,
                e.raw_text,
                THEME_SCHEMA,
                seed,
                limiter,
                cache,
                cache_stats,
                f"Theme discovery chunk {i}",
            )

        data, repair_usage, repaired = validate_or_repair_json(
            client,
            model,
            data,
            raw,
            THEME_SCHEMA,
            seed,
            limiter,
            cache,
            cache_stats,
            f"Theme discovery chunk {i}",
        )
        if repaired:
            usage["prompt_tokens"] += repair_usage.get("prompt_tokens", 0)
            usage["completion_tokens"] += repair_usage.get("completion_tokens", 0)

        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        all_themes.extend(data.get("major_themes", []))
        progress(40 + int((i / len(chunks)) * 40))

    merged = merge_theme_chunks(all_themes, on_notice=on_notice)
    progress(100)
    return {"major_themes": merged}, total_usage
