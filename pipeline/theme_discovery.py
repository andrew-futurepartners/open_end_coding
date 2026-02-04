from __future__ import annotations

import json
import re
import time
import hashlib
import difflib
import random
from concurrent.futures import ThreadPoolExecutor
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


DESTINATION_FUZZY_THRESHOLD = 0.95
DESTINATION_NORMALIZE_MODEL = "gpt-4o"
DESTINATION_BATCH_MAX_ITEMS = 10
DESTINATION_BATCH_TOKEN_CAP = 12_000
DESTINATION_BATCH_MAX_WORKERS = 6
DESTINATION_BATCH_TIMEOUT_SEC = 90
DESTINATION_NORMALIZE_TRUNCATE_CHARS = 200

DESTINATION_NORMALIZE_SYSTEM = (
    "You canonicalize destination responses. Use the question and guidance to decide how to normalize. "
    "If unsure, return a cleaned, title-cased version of the input. "
    "If the input is a non-response (e.g., n/a, none, prefer not to say), "
    "set is_non_response=true and label must be empty. Return English-only labels."
)

DESTINATION_NORMALIZE_USER_TEMPLATE = (
    "Schema version: v1\n"
    "Question: {question_text}\n"
    "Guidance: {guidance_text}\n"
    "Normalize each item to a canonical destination label.\n"
    "Rules:\n"
    "- Valid destinations include cities, states, parks, monuments, venues, attractions.\n"
    "- If ambiguous (e.g., 'Disney'), set is_general=true and label should be 'Disney (General)'.\n"
    "- Do NOT return empty label unless is_non_response=true.\n"
    "- If unsure, return cleaned, title-cased input.\n"
    "- Only mark non-response for explicit non-responses (n/a, none, prefer not, don't know).\n"
    "- Return English-only labels; no emojis.\n\n"
    "- Return one output item per input item and preserve idx.\n\n"
    "Examples:\n"
    "Input: \"Disney\"\n"
    "Output: {{\"label\":\"Disney (General)\",\"is_general\":true,\"is_non_response\":false}}\n"
    "Input: \"Disney land\"\n"
    "Output: {{\"label\":\"Disneyland\",\"is_general\":false,\"is_non_response\":false}}\n"
    "Input: \"LA\"\n"
    "Output: {{\"label\":\"Los Angeles, CA\",\"is_general\":false,\"is_non_response\":false}}\n"
    "Input: \"n/a\"\n"
    "Output: {{\"label\":\"\",\"is_general\":false,\"is_non_response\":true}}\n\n"
    "Items (JSON array):\n"
    "{items_json}\n"
)

DESTINATION_NORMALIZE_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "destination_normalization",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "idx": {"type": "integer"},
                            "label": {"type": "string"},
                            "is_general": {"type": "boolean"},
                            "is_non_response": {"type": "boolean"},
                        },
                        "required": ["idx", "label", "is_general", "is_non_response"],
                        "additionalProperties": True,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": True,
        },
        "strict": False,
    },
}

DESTINATION_ALIAS_MAP = {
    "la": "Los Angeles, CA",
    "nyc": "New York, NY",
    "sf": "San Francisco, CA",
    "dc": "Washington, DC",
    "sd": "San Diego, CA",
    "sj": "San Jose, CA",
    "lv": "Las Vegas, NV",
    "nola": "New Orleans, LA",
    "philly": "Philadelphia, PA",
    "atl": "Atlanta, GA",
}


def _destination_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _expand_destination_aliases(label: str) -> str:
    key = _destination_key(label)
    if not key:
        return ""
    return DESTINATION_ALIAS_MAP.get(key, label)


def _fuzzy_merge_destination(label: str, existing_labels: List[str]) -> str:
    if not existing_labels:
        return label
    key = _destination_key(label)
    if not key:
        return label
    best_label = label
    best_score = 0.0
    for existing in existing_labels:
        existing_key = _destination_key(existing)
        if not existing_key:
            continue
        score = difflib.SequenceMatcher(a=key, b=existing_key).ratio()
        if score > best_score:
            best_score = score
            best_label = existing
    if best_score >= DESTINATION_FUZZY_THRESHOLD:
        return best_label
    return label


def _clean_normalized_label(label: str) -> str:
    cleaned = (label or "").encode("ascii", errors="ignore").decode("ascii")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
    "garden",
    "square",
    "tower",
    "pier",
    "national park",
    "state park",
}


def _contains_attraction_keyword(label: str) -> bool:
    lowered = (label or "").lower()
    return any(keyword in lowered for keyword in _ATTRACTION_KEYWORDS)


def _is_more_specific(label: str, geocode_label: str) -> bool:
    if not label:
        return False
    if _contains_attraction_keyword(label) and not _contains_attraction_keyword(geocode_label):
        return True
    label_parts = [p for p in re.split(r"[,\s]+", label) if p]
    geo_parts = [p for p in re.split(r"[,\s]+", geocode_label or "") if p]
    return len(label_parts) > max(1, len(geo_parts))


_EXPLICIT_NON_RESPONSE = {
    "n/a",
    "na",
    "none",
    "no",
    "not applicable",
    "not sure",
    "dont know",
    "don't know",
    "unknown",
    "prefer not",
    "prefer not to say",
    "no answer",
    "nothing",
}


def _is_explicit_non_response(raw: str) -> bool:
    text = " ".join(str(raw or "").strip().lower().split())
    return text in _EXPLICIT_NON_RESPONSE


def _force_general_suffix(label: str) -> str:
    if not label:
        return ""
    if "(general)" in label.lower():
        return label
    return f"{label} (General)"


def _build_destination_batches(
    items: List[Dict[str, Any]],
    question_text: str,
    guidance_text: str,
    target_type: str,
) -> List[List[Dict[str, Any]]]:
    base_user = DESTINATION_NORMALIZE_USER_TEMPLATE.format(
        question_text=question_text,
        guidance_text=guidance_text,
        target_type=target_type,
        items_json="[]",
    )
    base_tokens = estimate_tokens(DESTINATION_NORMALIZE_SYSTEM + base_user)
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = base_tokens
    for item in items:
        item_str = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        item_tokens = estimate_tokens(item_str)
        if current and (len(current) >= DESTINATION_BATCH_MAX_ITEMS or (current_tokens + item_tokens) > DESTINATION_BATCH_TOKEN_CAP):
            batches.append(current)
            current = []
            current_tokens = base_tokens
        current.append(item)
        current_tokens += item_tokens
    if current:
        batches.append(current)
    return batches


def _normalize_destination_batch(
    client,
    items: List[Dict[str, Any]],
    question_text: str,
    guidance_text: str,
    target_type: str,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    seed: int,
) -> Dict[int, Dict[str, Any]]:
    items_json = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    user = DESTINATION_NORMALIZE_USER_TEMPLATE.format(
        question_text=question_text,
        guidance_text=guidance_text,
        target_type=target_type,
        items_json=items_json,
    )

    def _on_retry(_: str) -> None:
        time.sleep(random.uniform(0.05, 0.25))

    data, usage, raw, _ = retry_with_backoff(
        lambda: oai_json_completion(
            client,
            DESTINATION_NORMALIZE_MODEL,
            DESTINATION_NORMALIZE_SYSTEM,
            user,
            seed,
            response_schema=DESTINATION_NORMALIZE_SCHEMA,
            limiter=limiter,
            cache=cache,
            cache_stats=cache_stats,
            reasoning_effort="low",
            verbosity="low",
            reserve_output_tokens=800,
        ),
        max_retries=3,
        on_retry=_on_retry,
    )

    data, _, _ = validate_or_repair_json(
        client,
        DESTINATION_NORMALIZE_MODEL,
        data,
        raw,
        DESTINATION_NORMALIZE_SCHEMA,
        seed,
        limiter,
        cache,
        cache_stats,
        "Destination normalization",
    )
    results = {}
    items_out = data.get("items", []) if isinstance(data, dict) else []
    for item in items_out:
        idx = item.get("idx")
        if isinstance(idx, int):
            results[idx] = item
    if len(results) != len(items):
        raise ValueError(f"Normalization batch size mismatch: expected {len(items)}, got {len(results)}")
    return results


def _canonicalize_destinations_ai(
    client,
    raw_items: List[Dict[str, Any]],
    question_text: str,
    guidance_text: str,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    seed: int,
    normalization_map: Dict[str, str] | None = None,
) -> Dict[int, Dict[str, Any]]:
    if not raw_items:
        return {}

    sorted_items = sorted(raw_items, key=lambda x: str(x.get("text") or ""))
    batches = _build_destination_batches(sorted_items, question_text, guidance_text, "destination")
    results: Dict[int, Dict[str, Any]] = {}
    errors = 0

    def _run_batch(batch: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        return _normalize_destination_batch(client, batch, question_text, guidance_text, "destination", limiter, cache, cache_stats, seed)

    with ThreadPoolExecutor(max_workers=DESTINATION_BATCH_MAX_WORKERS) as executor:
        future_map = {executor.submit(_run_batch, batch): batch for batch in batches}
        for future, batch in future_map.items():
            try:
                batch_result = future.result(timeout=DESTINATION_BATCH_TIMEOUT_SEC)
                results.update(batch_result)
            except Exception:
                errors += 1
                for item in batch:
                    try:
                        single = _run_batch([item])
                        results.update(single)
                    except Exception:
                        continue

    if normalization_map is not None:
        samples = []
        for item in sorted_items:
            idx = item["idx"]
            raw_text = item["text"]
            normalized = results.get(idx, {}).get("label", "")
            is_non_response = results.get(idx, {}).get("is_non_response", False)
            mapped = "Non-response" if is_non_response else normalized
            normalization_map[str(raw_text)] = mapped
            if len(samples) < 10:
                samples.append({"raw": raw_text, "normalized": mapped, "notes": results.get(idx, {}).get("notes", "")})

    return results


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
    client=None,
    limiter: RateLimiter | None = None,
    cache: SQLiteCache | None = None,
    cache_stats: CacheStats | None = None,
    seed: int = 42,
    normalization_question_text: str | None = None,
    normalization_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    target_type = guidance_target_type if guidance_target_type and guidance_target_type != "auto" else _detect_target_type(guidance_text)
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    other_examples: List[str] = []
    other_count = 0
    if target_type in {"destination", "location", "city", "state", "country", "county", "brand", "adjective", "one_word", "general"}:
        prepared: List[Dict[str, Any]] = []
        resolved: List[Dict[str, Any]] = []
        debug_stats = {
            "total": 0,
            "ai_empty": 0,
            "ai_non_response": 0,
            "fallback_used": 0,
            "labels_added": 0,
        }
        override_count = 0
        override_samples: List[Dict[str, Any]] = []
        special_samples: List[Dict[str, Any]] = []
        for idx, (text, weight) in enumerate(zip(texts or [], freq or [])):
            raw = str(text or "").strip()
            if not raw:
                continue
            debug_stats["total"] += 1
            candidate = raw
            label = ""
            if target_type in {"location", "city", "state", "country", "county", "destination"}:
                candidate = _extract_location_candidate(raw)
                geocode_label = ""
                if target_type == "destination":
                    candidate = _expand_destination_aliases(candidate)
                if normalize_locations and geocode_user_agent:
                    if target_type == "country":
                        geocode_label, _ = normalize_country(
                            candidate,
                            user_agent=geocode_user_agent,
                            cache=geocode_cache,
                            limiter=geocode_limiter,
                        )
                    elif target_type == "state":
                        geocode_label, _ = normalize_state(
                            candidate,
                            user_agent=geocode_user_agent,
                            cache=geocode_cache,
                            limiter=geocode_limiter,
                        )
                    elif target_type == "county":
                        geocode_label, _ = normalize_county(
                            candidate,
                            user_agent=geocode_user_agent,
                            cache=geocode_cache,
                            limiter=geocode_limiter,
                        )
                    elif target_type == "city":
                        geocode_label, _ = normalize_city(
                            candidate,
                            user_agent=geocode_user_agent,
                            cache=geocode_cache,
                            limiter=geocode_limiter,
                        )
                    else:
                        geocode_label, _ = normalize_location(
                            candidate,
                            user_agent=geocode_user_agent,
                            cache=geocode_cache,
                            limiter=geocode_limiter,
                        )
                if target_type != "destination":
                    label = geocode_label or label
            elif target_type == "adjective":
                label = _normalize_adjective(raw)
            elif target_type == "one_word":
                label = _normalize_one_word(raw)
            elif target_type == "brand":
                label = _normalize_brand_destination(raw)
            else:
                label = _normalize_value_label(raw)

            if target_type == "destination" and label:
                resolved.append({
                    "idx": idx,
                    "raw": raw,
                    "weight": int(weight or 0),
                    "label": label,
                })
            else:
                if target_type == "destination":
                    ai_text_source = "candidate"
                    ai_text = candidate
                    fallback_label = label or candidate
                else:
                    ai_text_source = "label" if label else ("geocode" if geocode_label else "candidate")
                    ai_text = label or geocode_label or candidate
                    fallback_label = label or geocode_label or candidate
                ai_text = (ai_text or "")[:DESTINATION_NORMALIZE_TRUNCATE_CHARS]
                prepared.append({
                    "idx": idx,
                    "raw": raw,
                    "weight": int(weight or 0),
                    "ai_text": ai_text,
                    "ai_text_source": ai_text_source,
                    "fallback_label": fallback_label,
                    "geocode_label": geocode_label,
                })

        norm_results: Dict[int, Dict[str, Any]] = {}
        if prepared and client and limiter and cache and cache_stats:
            items = [{"idx": item["idx"], "text": item["ai_text"]} for item in prepared]
            norm_results = _canonicalize_destinations_ai(
                client,
                items,
                normalization_question_text or guidance_text or "Destination normalization",
                guidance_text or "No additional guidance.",
                limiter,
                cache,
                cache_stats,
                seed,
                normalization_map=normalization_map,
            )
            non_empty = 0
            for item in prepared:
                result = norm_results.get(item["idx"], {})
                if _clean_normalized_label(result.get("label", "") or ""):
                    non_empty += 1
            if prepared and (non_empty / max(1, len(prepared)) < 0.2):
                norm_results = {}

        for item in resolved:
            raw = item["raw"]
            weight = item["weight"]
            label = _clean_normalized_label(item.get("label", ""))
            if not label:
                continue
            label = _clean_normalized_label(_normalize_brand_destination(label))
            counts[label] = counts.get(label, 0) + int(weight or 0)
            if label not in examples:
                examples[label] = []
            if len(examples[label]) < 3:
                examples[label].append(label)

        for item in prepared:
            raw = item["raw"]
            weight = item["weight"]
            ai_text = item["ai_text"]
            result = norm_results.get(item["idx"], {})
            label = _clean_normalized_label(result.get("label", "") or "")
            is_general = bool(result.get("is_general", False))
            is_non_response = bool(result.get("is_non_response", False)) and _is_explicit_non_response(raw)
            geocode_label = _clean_normalized_label(item.get("geocode_label", "") or "")

            if _destination_key(label) == "disney":
                is_general = True
                label = "Disney"

            if is_non_response:
                debug_stats["ai_non_response"] += 1
                other_count += int(weight or 0)
                if len(other_examples) < 3:
                    other_examples.append("Non-response")
                continue

            if not label:
                debug_stats["ai_empty"] += 1
                fallback = _clean_normalized_label(item.get("fallback_label", "") or "")
                if not fallback:
                    fallback = _clean_normalized_label(_normalize_brand_destination(ai_text))
                label = fallback
                debug_stats["fallback_used"] += 1
                if not label:
                    if soft_prefer:
                        other_count += int(weight or 0)
                        if len(other_examples) < 3 and raw not in other_examples:
                            other_examples.append(raw)
                    continue

            label = _clean_normalized_label(_normalize_brand_destination(label))
            more_specific = _is_more_specific(label, geocode_label) if geocode_label else None
            if geocode_label and (not label or label == geocode_label):
                override_count += 1
                if len(override_samples) < 8:
                    override_samples.append({
                        "raw": raw,
                        "ai_label": label,
                        "geocode_label": geocode_label,
                        "is_general": is_general,
                        "ai_has_attraction_kw": _contains_attraction_keyword(label),
                        "geo_has_attraction_kw": _contains_attraction_keyword(geocode_label),
                    })
                label = geocode_label
            if is_general or "(general)" in label.lower():
                label = _force_general_suffix(label)
            elif target_type == "destination":
                label = _fuzzy_merge_destination(label, list(counts.keys()))
            if any(k in raw.lower() for k in ("disney", "yosemite", "statue of liberty", "area 51")) and len(special_samples) < 12:
                special_samples.append({
                    "raw": raw,
                    "ai_label": label,
                    "geocode_label": geocode_label,
                    "is_general": is_general,
                    "more_specific": more_specific,
                    "ai_text_source": item.get("ai_text_source"),
                    "ai_text": item.get("ai_text"),
                    "fallback_label": item.get("fallback_label"),
                })

            counts[label] = counts.get(label, 0) + int(weight or 0)
            debug_stats["labels_added"] += 1
            if label not in examples:
                examples[label] = []
            if len(examples[label]) < 3:
                examples[label].append(label if target_type == "destination" else raw)

        if len(counts) < 3:
            fallback_labels = {}
            for item in prepared:
                raw = item["raw"]
                if _is_explicit_non_response(raw):
                    continue
                fallback = _clean_normalized_label(item.get("fallback_label", "") or "")
                if not fallback:
                    continue
                fallback_labels[fallback] = fallback_labels.get(fallback, 0) + int(item["weight"] or 0)
            if fallback_labels:
                counts = fallback_labels
                examples = {k: [k] for k in list(counts.keys())[:50]}
        if not counts:
            for item in prepared:
                raw = _clean_normalized_label(item["raw"])
                if not raw or _is_explicit_non_response(raw):
                    continue
                counts[raw] = counts.get(raw, 0) + int(item["weight"] or 0)
                examples.setdefault(raw, []).append(raw)
    else:
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
            elif target_type == "brand":
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
    max_subthemes = None
    if max_subthemes and len(ordered) > max_subthemes:
        if any(lbl == "Other" for lbl, _ in ordered):
            ordered = [item for item in ordered if item[0] != "Other"][: max_subthemes - 1] + [
                ("Other", counts.get("Other", 0))
            ]
        else:
            ordered = ordered[:max_subthemes]
    total = sum(cnt for _, cnt in ordered) or 1

    if target_type in {"city", "state", "country", "county", "location", "destination"}:
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
        elif target_type == "destination":
            major_label = "Destinations"
            major_def = "Destination categories derived from responses."
        else:
            major_label = "Locations"
            major_def = "Location categories derived from responses."
    elif target_type == "brand":
        major_label = "Brands"
        major_def = "Brand categories derived from responses."
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
    normalization_question_text: str | None = None,
    normalization_map: Dict[str, str] | None = None,
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
    if guidance_text:
        enhanced_prompt += f"\n\nGuidance (must follow):\n{guidance_text}\n"
        enhanced_prompt += (
            "Guidance rules:\n"
            "- Treat guidance as hard constraints; do not invent unrelated themes.\n"
            "- If a response does not fit the guidance, use a catch-all Other or Non-answer bucket.\n"
            "- Prefer concrete categories implied by guidance (e.g., cities, brands, destinations).\n"
        )

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
    if guidance_target_type and guidance_target_type != "auto":
        guided = _build_guided_codeframe(
            texts,
            freq,
            guidance_text or "",
            normalize_locations=normalize_locations,
            geocode_user_agent=geocode_user_agent,
            geocode_cache=geocode_cache or get_default_geocode_cache(),
            geocode_limiter=geocode_limiter or get_default_geocode_limiter(),
            guidance_target_type=guidance_target_type,
            soft_prefer=guidance_soft_prefer,
            client=client,
            limiter=limiter,
            cache=cache,
            cache_stats=cache_stats,
            seed=seed,
            normalization_question_text=normalization_question_text,
            normalization_map=normalization_map,
        )
        merged = guided
        progress(100)
        return guided, total_usage
    elif guidance_text and _guidance_mismatch(guidance_text, merged):
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
            client=client,
            limiter=limiter,
            cache=cache,
            cache_stats=cache_stats,
            seed=seed,
            normalization_question_text=normalization_question_text,
            normalization_map=normalization_map,
        )
        progress(100)
        return guided, total_usage
    progress(100)
    return {"major_themes": merged}, total_usage
