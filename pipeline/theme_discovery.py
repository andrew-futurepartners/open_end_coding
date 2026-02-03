from __future__ import annotations

import json
import re
import time
import hashlib
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
from pipeline.geocode import (
    GeoCache,
    GeoRateLimiter,
    get_default_geocode_cache,
    get_default_geocode_limiter,
    normalize_city,
    normalize_county,
    normalize_country,
    normalize_location,
    normalize_state,
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


def _guidance_expected_tokens(guidance_text: str) -> List[str]:
    text = (guidance_text or "").lower()
    tokens: List[str] = []
    if any(k in text for k in ["city", "cities", "area", "areas", "location", "region", "state", "country", "county"]):
        tokens.extend(["city", "area", "location", "region", "state", "country", "county", "neighborhood"])
    if "brand" in text or "brands" in text:
        tokens.append("brand")
    if "destination" in text or "destinations" in text:
        tokens.append("destination")
    if "product" in text or "products" in text:
        tokens.append("product")
    if "service" in text or "services" in text:
        tokens.append("service")
    if "adjective" in text or "descriptor" in text:
        tokens.append("adjective")
    if "one word" in text or "single word" in text:
        tokens.append("oneword")
    return sorted(set(tokens))


def _detect_target_type(guidance_text: str | None) -> str:
    text = (guidance_text or "").lower()
    if any(k in text for k in ["one word", "single word", "one-word"]):
        return "one_word"
    if any(k in text for k in ["adjective", "descriptor", "descriptive word"]):
        return "adjective"
    if any(k in text for k in ["country", "countries"]):
        return "country"
    if any(k in text for k in ["state", "states", "us state", "u.s. state"]):
        return "state"
    if "county" in text or "counties" in text:
        return "county"
    if any(k in text for k in ["brand", "brands"]):
        return "brand"
    if any(k in text for k in ["destination", "destinations"]):
        return "destination"
    if any(k in text for k in ["location", "area", "areas", "region", "neighborhood", "neighbourhood"]):
        return "location"
    if any(k in text for k in ["city", "cities"]):
        return "city"
    return "general"


def _guidance_mismatch(guidance_text: str, merged: List[Dict[str, Any]]) -> bool:
    expected = _guidance_expected_tokens(guidance_text)
    if not expected:
        return False
    labels = []
    for major in merged or []:
        labels.append(str(major.get("label", "")).lower())
        for sub in major.get("subs", []) or []:
            labels.append(str(sub.get("label", "")).lower())
    for token in expected:
        if any(token in lbl for lbl in labels):
            return False
    return True


def _normalize_value_label(value: str) -> str:
    s = re.sub(r"\s+", " ", str(value or "").strip())
    s = s.strip(" .;:,")
    if not s:
        return ""
    letters = sum(1 for c in s if c.isalpha())
    if letters >= max(1, len(s) // 3):
        parts = []
        for p in s.split(" "):
            if p.isupper() or any(ch.isdigit() for ch in p):
                parts.append(p)
            else:
                parts.append(p.capitalize())
        s = " ".join(parts)
    return s


def _extract_location_candidate(text: str) -> str:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return ""
    lowered = s.lower()
    for token in (" lived in ", " live in ", " moved to ", " from ", " in ", " near ", " around ", " at "):
        if token in lowered:
            s = s[lowered.rfind(token) + len(token):].strip()
            break
    s = re.sub(r"^[^A-Za-z]+", "", s).strip()
    s = re.sub(r"\s*[\.\!\?]+$", "", s).strip()
    return s or str(text or "").strip()


def _normalize_adjective(text: str) -> str:
    s = " ".join(str(text or "").strip().split())
    if not s:
        return ""
    s = re.sub(r"[^\w\s-]", "", s)
    parts = s.split()
    if len(parts) == 0:
        return ""
    if len(parts) > 1:
        if parts[0].lower() in {"very", "really", "quite", "so", "super", "extremely"}:
            parts = parts[1:]
    if len(parts) != 1:
        return ""
    return parts[0].lower()


def _normalize_one_word(text: str) -> str:
    s = re.sub(r"[^\w-]", " ", str(text or "").strip())
    parts = [p for p in s.split() if p]
    if len(parts) != 1:
        return ""
    return parts[0]


def _normalize_brand_destination(text: str) -> str:
    return _normalize_value_label(text)


def _build_guided_codeframe(
    texts: List[str],
    freq: List[int],
    guidance_text: str,
    normalize_locations: bool = False,
    geocode_user_agent: str | None = None,
    geocode_cache: GeoCache | None = None,
    geocode_limiter: GeoRateLimiter | None = None,
    guidance_target_type: str | None = None,
    soft_prefer: bool = True,
) -> Dict[str, Any]:
    target_type = guidance_target_type if guidance_target_type and guidance_target_type != "auto" else _detect_target_type(guidance_text)
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    other_examples: List[str] = []
    other_count = 0
    for text, weight in zip(texts or [], freq or []):
        raw = str(text or "").strip()
        if not raw:
            continue
        label = ""
        if target_type in {"city", "state", "country", "county", "location"}:
            candidate = _extract_location_candidate(raw)
            if normalize_locations and geocode_user_agent:
                if target_type == "country":
                    label, _ = normalize_country(
                        candidate,
                        user_agent=geocode_user_agent,
                        cache=geocode_cache,
                        limiter=geocode_limiter,
                    )
                elif target_type == "state":
                    label, _ = normalize_state(
                        candidate,
                        user_agent=geocode_user_agent,
                        cache=geocode_cache,
                        limiter=geocode_limiter,
                    )
                elif target_type == "county":
                    label, _ = normalize_county(
                        candidate,
                        user_agent=geocode_user_agent,
                        cache=geocode_cache,
                        limiter=geocode_limiter,
                    )
                elif target_type == "city":
                    label, _ = normalize_city(
                        candidate,
                        user_agent=geocode_user_agent,
                        cache=geocode_cache,
                        limiter=geocode_limiter,
                    )
                else:
                    label, _ = normalize_location(
                        candidate,
                        user_agent=geocode_user_agent,
                        cache=geocode_cache,
                        limiter=geocode_limiter,
                    )
            if not label:
                label = _normalize_value_label(candidate)
        elif target_type == "adjective":
            label = _normalize_adjective(raw)
        elif target_type == "one_word":
            label = _normalize_one_word(raw)
        elif target_type in {"brand", "destination"}:
            label = _normalize_brand_destination(raw)
        else:
            label = _normalize_value_label(raw)
        if not label:
            if soft_prefer:
                other_count += int(weight or 0)
                if len(other_examples) < 3 and raw not in other_examples:
                    other_examples.append(raw)
            continue
        counts[label] = counts.get(label, 0) + int(weight or 0)
        if label not in examples:
            examples[label] = []
        if len(examples[label]) < 3 and raw not in examples[label]:
            examples[label].append(raw)

    if other_count > 0:
        counts["Other"] = counts.get("Other", 0) + other_count
        if "Other" not in examples:
            examples["Other"] = []
        for ex in other_examples[:3]:
            if ex not in examples["Other"]:
                examples["Other"].append(ex)

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    max_subthemes = 50
    if len(ordered) > max_subthemes:
        if any(lbl == "Other" for lbl, _ in ordered):
            ordered = [item for item in ordered if item[0] != "Other"][: max_subthemes - 1] + [
                ("Other", counts.get("Other", 0))
            ]
        else:
            ordered = ordered[:max_subthemes]
    total = sum(cnt for _, cnt in ordered) or 1

    if target_type in {"city", "state", "country", "county", "location"}:
        if target_type == "country":
            major_label = "Countries"
            major_def = "Country categories derived from responses."
        elif target_type == "state":
            major_label = "States"
            major_def = "State categories derived from responses."
        elif target_type == "county":
            major_label = "Counties"
            major_def = "County categories derived from responses."
        elif target_type == "city":
            major_label = "Cities"
            major_def = "City categories derived from responses."
        else:
            major_label = "Locations"
            major_def = "Location categories derived from responses."
    elif target_type == "brand":
        major_label = "Brands"
        major_def = "Brand categories derived from responses."
    elif target_type == "destination":
        major_label = "Destinations"
        major_def = "Destination categories derived from responses."
    elif target_type == "adjective":
        major_label = "Descriptors"
        major_def = "Adjective descriptors derived from responses."
    elif target_type == "one_word":
        major_label = "One-word Responses"
        major_def = "Single-word responses derived from the question."
    else:
        major_label = "Guided Responses"
        major_def = "Categories derived from responses under the provided guidance."

    subs = []
    for idx, (label, cnt) in enumerate(ordered, start=1):
        subs.append({
            "id": f"T1.{idx}",
            "label": label,
            "definition": f"Responses indicating {label}.",
            "approx_pct": float(cnt) / float(total),
            "examples": examples.get(label, [])[:3],
        })

    theme_dict = {
        "major_themes": [
            {
                "id": "T1",
                "label": major_label,
                "definition": major_def,
                "approx_pct": 1.0,
                "subs": subs,
            },
            {
                "id": "T999",
                "label": "Non-answer",
                "definition": "Responses that do not provide a meaningful answer.",
                "approx_pct": 0.0,
                "subs": [
                    {"id": "T999.1", "label": "Refusal", "definition": "Refuses to answer.", "approx_pct": 0.0, "examples": []},
                    {"id": "T999.2", "label": "Don't know", "definition": "Does not know.", "approx_pct": 0.0, "examples": []},
                    {"id": "T999.3", "label": "Nonsense", "definition": "Unintelligible or nonsensical.", "approx_pct": 0.0, "examples": []},
                    {"id": "T999.4", "label": "Spam", "definition": "Spam or irrelevant text.", "approx_pct": 0.0, "examples": []},
                    {"id": "T999.5", "label": "Not applicable", "definition": "Not applicable.", "approx_pct": 0.0, "examples": []},
                ],
            },
        ]
    }
    return theme_dict


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
    guidance_text: str | None = None,
    normalize_locations: bool = False,
    geocode_user_agent: str | None = None,
    geocode_cache: GeoCache | None = None,
    geocode_limiter: GeoRateLimiter | None = None,
    guidance_target_type: str | None = None,
    guidance_soft_prefer: bool = True,
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
    # #region agent log
    try:
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H1",
                "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:entry",
                "message": "Theme discovery entry",
                "data": {
                    "text_count": len(texts or []),
                    "guidance_len": len((guidance_text or "").strip()),
                    "question_context_type": (question_context or {}).get("type"),
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion
    progress(10)
    filtered_data = [
        {"text": t, "weight": int(w)}
        for t, w in zip(texts, freq)
        if str(t or "").strip() != ""
    ]
    # #region agent log
    try:
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H2",
                "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:filtered",
                "message": "Filtered data ready",
                "data": {
                    "filtered_count": len(filtered_data),
                    "force_chunking": True,
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion
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
    if guidance_text:
        enhanced_prompt += f"\n\nGuidance (must follow):\n{guidance_text}\n"
        enhanced_prompt += (
            "Guidance rules:\n"
            "- Treat guidance as hard constraints; do not invent unrelated themes.\n"
            "- If a response does not fit the guidance, use a catch-all Other or Non-answer bucket.\n"
            "- Prefer concrete categories implied by guidance (e.g., cities, brands, destinations).\n"
        )
    # #region agent log
    try:
        guidance_hash = ""
        if guidance_text:
            guidance_hash = hashlib.sha256(guidance_text.strip().encode("utf-8")).hexdigest()
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H16",
                "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:prompt",
                "message": "Enhanced prompt built",
                "data": {
                    "guidance_in_prompt": bool(guidance_text and guidance_text.strip()),
                    "guidance_hash": guidance_hash,
                    "prompt_len": len(enhanced_prompt),
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion

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
    # #region agent log
    try:
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H10",
                "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:chunking",
                "message": "Chunking theme discovery",
                "data": {
                    "chunk_count": len(chunks),
                    "filtered_count": len(filtered_data),
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion

    for i, chunk in enumerate(chunks, start=1):
        payload = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        user = enhanced_prompt + "\n\nWeighted responses (JSON array):\n" + payload
        # #region agent log
        try:
            with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H11",
                    "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:chunk_prompt",
                    "message": "Chunk prompt ready",
                    "data": {
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "user_len": len(user),
                        "guidance_in_prompt": ("Guidance (must follow):" in user) or ("GUIDANCE:" in user),
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion

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
        # #region agent log
        try:
            majors = data.get("major_themes", []) if isinstance(data, dict) else []
            with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H12",
                    "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:chunk_result",
                    "message": "Chunk themes received",
                    "data": {
                        "chunk_index": i,
                        "major_count": len(majors),
                        "major_labels": [m.get("label", "") for m in majors][:8],
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        if repaired:
            usage["prompt_tokens"] += repair_usage.get("prompt_tokens", 0)
            usage["completion_tokens"] += repair_usage.get("completion_tokens", 0)

        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        all_themes.extend(data.get("major_themes", []))
        progress(40 + int((i / len(chunks)) * 40))

    merged = merge_theme_chunks(all_themes, on_notice=on_notice)
    if guidance_text and _guidance_mismatch(guidance_text, merged):
        guided = _build_guided_codeframe(
            texts,
            freq,
            guidance_text,
            normalize_locations=normalize_locations,
            geocode_user_agent=geocode_user_agent,
            geocode_cache=geocode_cache or get_default_geocode_cache(),
            geocode_limiter=geocode_limiter or get_default_geocode_limiter(),
            guidance_target_type=guidance_target_type,
            soft_prefer=guidance_soft_prefer,
        )
        # #region agent log
        try:
            with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H17",
                    "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:guided_fallback",
                    "message": "Guided fallback applied",
                    "data": {
                        "expected_tokens": _guidance_expected_tokens(guidance_text),
                        "major_count": len(guided.get("major_themes", [])),
                        "sub_count": len((guided.get("major_themes", [{}])[0].get("subs", [])) if guided.get("major_themes") else []),
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        progress(100)
        return guided, total_usage
    # #region agent log
    try:
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H9",
                "location": "pipeline/theme_discovery.py:build_theme_frame_with_progress:merged",
                "message": "Merged theme labels",
                "data": {
                    "major_labels": [m.get("label", "") for m in (merged or [])][:10],
                    "major_count": len(merged or []),
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion
    progress(100)
    return {"major_themes": merged}, total_usage
