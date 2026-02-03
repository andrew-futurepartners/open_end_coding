from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple, Callable

import numpy as np
from openai import OpenAI

from assignment_utils import (
    build_assignments_list,
    make_assignment_decision_schema,
    normalize_assignment_result,
    normalize_candidate_ids,
    stable_candidate_key,
)
from candidate_retrieval import build_subtheme_records, embed_texts, get_candidate_ids, theme_signature
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
from pipeline.utils import chunk_data, estimate_tokens, safe_prompt_token_budget


ASSIGNMENT_STAGE2_SYSTEM = (
    "You are a meticulous qualitative coder. You must choose from the provided candidate sub-themes only. "
    "Do not invent new themes."
)

ASSIGNMENT_STAGE2_USER_TEMPLATE = (
    """
You will assign the following responses to the provided candidate sub-themes ONLY.

Rules:
- Choose 1..{max_codes} subtheme IDs from the candidate list.
- If no candidate fits, choose the closest candidate but set decision="needs_review" and confidence <= 0.4.
- Provide a short rationale (30-50 words max). No chain-of-thought.
- If confidence < {low_thresh}, decision must be "needs_review".

Return JSON in this exact format:
{{
  "results": [
    {{
      "idx": <row index integer>,
      "subtheme_ids": ["T1.2"],
      "confidence": 0.87,
      "decision": "ok",
      "rationale": "Short justification."
    }}
  ]
}}

Candidate sub-themes:
{candidate_json}

Guidance (must follow):
{guidance_text}

Responses:
{responses_json}
"""
)


def allowed_subtheme_ids(theme_dict: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for major in theme_dict.get("major_themes", []):
        for sub in (major.get("subs") or []):
            sid = sub.get("id")
            if sid:
                ids.append(sid)
    return sorted(set(ids))


def ensure_nonanswer_theme(theme_dict: Dict[str, Any]) -> Dict[str, Any]:
    def _norm_alpha(x: str) -> str:
        return re.sub(r"[^a-z]+", "", (x or "").lower())

    majors = theme_dict.get("major_themes", []) or []
    for m in majors:
        if _norm_alpha(m.get("label", "")) == "nonanswer":
            return theme_dict

    nonanswer = {
        "id": "T999",
        "label": "Non-answer",
        "definition": "Responses that are refusals, don't know, nonsense, spam, or not applicable.",
        "approx_pct": 0.0,
        "subs": [
            {"id": "T999.1", "label": "Refusal", "definition": "Respondent refused to answer.", "approx_pct": 0.0, "examples": []},
            {"id": "T999.2", "label": "Don't know", "definition": "Respondent does not know.", "approx_pct": 0.0, "examples": []},
            {"id": "T999.3", "label": "Nonsense", "definition": "Uninterpretable response.", "approx_pct": 0.0, "examples": []},
            {"id": "T999.4", "label": "Spam", "definition": "Spam or irrelevant response.", "approx_pct": 0.0, "examples": []},
            {"id": "T999.5", "label": "Not applicable", "definition": "Not applicable to respondent.", "approx_pct": 0.0, "examples": []},
        ],
    }
    theme_dict["major_themes"] = majors + [nonanswer]
    return theme_dict


def assign_codes_two_stage(
    client: OpenAI,
    model: str,
    theme_dict: Dict[str, Any],
    rows: List[Dict[str, Any]],
    max_codes: int,
    seed: int,
    low_thresh: float,
    top_k: int,
    embedding_model: str,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    on_status: Callable[[str], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
    guidance_text: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    debug_enabled = os.getenv("ASSIGNMENT_DEBUG_LOG", "").lower() in ("1", "true", "yes")
    debug_buffer: List[Dict[str, Any]] = []

    def flush_debug() -> None:
        if not debug_enabled or not debug_buffer:
            return
        try:
            with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
                for event in debug_buffer:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            debug_buffer.clear()
        except Exception:
            debug_buffer.clear()

    def record_debug(event: Dict[str, Any]) -> None:
        if not debug_enabled:
            return
        debug_buffer.append(event)
        if len(debug_buffer) >= 25:
            flush_debug()

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    def progress(val: int) -> None:
        if on_progress:
            on_progress(val)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}
    status("Preparing two-stage assignment...")
    # #region agent log
    record_debug({
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": "H4",
        "location": "pipeline/assignment.py:assign_codes_two_stage:entry",
        "message": "Assignment entry",
        "data": {
            "rows": len(rows or []),
            "guidance_len": len((guidance_text or "").strip()),
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion
    progress(5)

    theme_dict = ensure_nonanswer_theme(theme_dict)

    def _norm(x: str) -> str:
        return re.sub(r"[^a-z]+", "", (x or "").lower())

    nonanswer_major = next(
        (m for m in theme_dict.get("major_themes", []) or [] if _norm(m.get("label", "")) == "nonanswer"), None)
    nonanswer_sub_ids = {_norm(s.get("label", "")): s.get("id") for s in
                         (nonanswer_major.get("subs", []) if nonanswer_major else [])}

    default_nonanswer_id = (
            nonanswer_sub_ids.get("notapplicable")
            or nonanswer_sub_ids.get("dontknow")
            or next(iter(nonanswer_sub_ids.values()), None)
    )

    def _pick_nonanswer_id(text: str) -> str:
        t = (text or "").strip().lower()
        if not t:
            key = "notapplicable"
        elif any(p in t for p in ["prefer not", "rather not", "no comment", "refuse", "pass"]):
            key = "refusal"
        elif any(p in t for p in ["don't know", "dont know", "idk", "not sure", "unsure", "unknown"]):
            key = "dontknow"
        elif any(p in t for p in ["spam", "http", "www."]):
            key = "spam"
        else:
            key = "notapplicable"

        return nonanswer_sub_ids.get(key) or default_nonanswer_id or "T999.5"

    status("Deduplicating responses...")
    progress(10)
    text_to_indices: Dict[str, List[int]] = {}
    for row in rows:
        text = row.get("text", "").strip()
        idx = row.get("idx")
        if idx is None:
            continue
        text_to_indices.setdefault(text, []).append(idx)

    unique_texts = list(text_to_indices.keys())
    all_subtheme_ids = allowed_subtheme_ids(theme_dict)

    text_to_assignment: Dict[str, Dict[str, Any]] = {}
    substantive_texts: List[str] = []
    for text in unique_texts:
        if str(text or "").strip() == "":
            text_to_assignment[text] = {
                "subtheme_ids": [],
                "confidence": 0.0,
                "decision": "ok",
                "rationale": "Skipped: empty",
            }
        else:
            substantive_texts.append(text)

    # #region agent log
    targets = [t for t in substantive_texts if "orange county" in (t or "").lower()]
    if targets:
        record_debug({
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": "H26",
            "location": "pipeline/assignment.py:assign_codes_two_stage:targets",
            "message": "Target texts for assignment",
            "data": {
                "texts": targets[:3],
                "target_count": len(targets),
            },
            "timestamp": int(time.time() * 1000),
        })
    # #endregion

    if substantive_texts:
        status("Building candidate shortlist (embeddings)...")
        progress(20)

        subtheme_records = build_subtheme_records(theme_dict)
        subtheme_ids = [r["id"] for r in subtheme_records if r.get("id")]
        subtheme_texts = [r["text"] for r in subtheme_records]

        cache_key = f"embeddings::{theme_signature(theme_dict)}::{embedding_model}"
        cached = cache.get(cache_key)
        if cached and cached.get("parsed_json"):
            subtheme_embeddings = np.array(cached["parsed_json"]["embeddings"], dtype=float)
        else:
            subtheme_embeddings_list, subtheme_tokens = embed_texts(client, embedding_model, subtheme_texts)
            subtheme_embeddings = np.array(subtheme_embeddings_list, dtype=float)
            cache.set(cache_key, embedding_model, "embeddings", json.dumps({"embeddings": subtheme_embeddings_list}), {"embeddings": subtheme_embeddings_list}, {})
            total_usage["embedding_tokens"] += subtheme_tokens

        response_embeddings_list, response_tokens = embed_texts(client, embedding_model, substantive_texts)
        total_usage["embedding_tokens"] += response_tokens
        response_embeddings = np.array(response_embeddings_list, dtype=float)

        candidate_lists = get_candidate_ids(response_embeddings, subtheme_embeddings, subtheme_ids, top_k=top_k)

        for text, candidates in zip(substantive_texts, candidate_lists):
            normalized_candidates = normalize_candidate_ids(candidates)
            if not normalized_candidates:
                normalized_candidates = [default_nonanswer_id] if default_nonanswer_id else []
            text_to_assignment[text] = {"candidate_ids": normalized_candidates}
            # #region agent log
            if "orange county" in (text or "").lower():
                record_debug({
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H27",
                    "location": "pipeline/assignment.py:assign_codes_two_stage:candidates",
                    "message": "Candidate shortlist for target",
                    "data": {
                        "text": text,
                        "candidate_ids": normalized_candidates[:8],
                    },
                    "timestamp": int(time.time() * 1000),
                })
            # #endregion

    if substantive_texts:
        status("Assigning with candidate-only prompts...")
        progress(40)

        candidate_group: Dict[Tuple[str, ...], List[str]] = {}
        for text in substantive_texts:
            candidate_ids = text_to_assignment[text].get("candidate_ids", [])
            key = stable_candidate_key(candidate_ids)
            candidate_group.setdefault(key, []).append(text)

        completed_groups = 0
        total_groups = max(1, len(candidate_group))
        subtheme_by_id = {r["id"]: r for r in subtheme_records}

        guidance_value = guidance_text or "None"

        for group_index, (candidate_key, texts_in_group) in enumerate(candidate_group.items(), start=1):
            candidate_ids = list(candidate_key) or [default_nonanswer_id or (all_subtheme_ids[0] if all_subtheme_ids else "T999.5")]
            candidate_payload = [
                {
                    "id": cid,
                    "label": subtheme_by_id.get(cid, {}).get("label", ""),
                    "definition": subtheme_by_id.get(cid, {}).get("definition", ""),
                    "example": subtheme_by_id.get(cid, {}).get("example", ""),
                }
                for cid in candidate_ids
            ]

            candidate_json = json.dumps(candidate_payload, ensure_ascii=False, separators=(",", ":"))
            user_prefix = ASSIGNMENT_STAGE2_USER_TEMPLATE.format(
                max_codes=max_codes,
                low_thresh=low_thresh,
                candidate_json=candidate_json,
                guidance_text=guidance_value,
                responses_json="__RESPONSES__",
            )

            response_items = [{"idx": i, "text": t} for i, t in enumerate(texts_in_group)]
            candidate_tokens = estimate_tokens(candidate_json, model=model)
            budget = safe_prompt_token_budget(model, reserve_output_tokens=4_000)
            overhead_tokens = 1600
            response_budget = max(2000, budget - candidate_tokens - overhead_tokens)
            response_chunks = chunk_data(response_items, max_tokens=response_budget, model=model)

            schema = make_assignment_decision_schema(candidate_ids, max_codes=max_codes)

            for chunk_index, chunk in enumerate(response_chunks, start=1):
                responses_json = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
                user = user_prefix.replace("__RESPONSES__", responses_json)
                # #region agent log
                record_debug({
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H5",
                    "location": "pipeline/assignment.py:assign_codes_two_stage:prompt",
                    "message": "Assignment prompt built",
                    "data": {
                        "candidate_count": len(candidate_ids),
                        "guidance_len": len(guidance_value.strip()),
                        "prompt_len": len(user),
                    },
                    "timestamp": int(time.time() * 1000),
                })
                # #endregion

                def make_request():
                    return oai_json_completion(
                        client,
                        model,
                        ASSIGNMENT_STAGE2_SYSTEM,
                        user,
                        seed,
                        response_schema=schema,
                        limiter=limiter,
                        cache=cache,
                        cache_stats=cache_stats,
                        reasoning_effort="minimal",
                        verbosity="low",
                        reserve_output_tokens=2_000,
                    )

                try:
                    hits_before = cache_stats.hits
                    misses_before = cache_stats.misses
                    call_start = time.perf_counter()
                    data, usage, raw, cached = retry_with_backoff(make_request)
                    elapsed_ms = int((time.perf_counter() - call_start) * 1000)
                    record_debug({
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "H5a",
                        "location": "pipeline/assignment.py:assign_codes_two_stage:llm_call",
                        "message": "Assignment LLM call metrics",
                        "data": {
                            "group_index": group_index,
                            "chunk_index": chunk_index,
                            "chunk_size": len(chunk),
                            "elapsed_ms": elapsed_ms,
                            "cache_hit": bool(cached),
                            "cache_hits_delta": cache_stats.hits - hits_before,
                            "cache_misses_delta": cache_stats.misses - misses_before,
                        },
                        "timestamp": int(time.time() * 1000),
                    })
                except JsonParseError as e:
                    data, usage, raw = repair_json_to_schema(
                        client,
                        model,
                        e.raw_text,
                        schema,
                        seed,
                        limiter,
                        cache,
                        cache_stats,
                        "Assignment stage 2",
                    )

                data, repair_usage, repaired = validate_or_repair_json(
                    client,
                    model,
                    data,
                    raw,
                    schema,
                    seed,
                    limiter,
                    cache,
                    cache_stats,
                    "Assignment stage 2",
                )
                if repaired:
                    usage["prompt_tokens"] += repair_usage.get("prompt_tokens", 0)
                    usage["completion_tokens"] += repair_usage.get("completion_tokens", 0)

                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

                results = data.get("results") if isinstance(data, dict) else []
                by_idx = {r.get("idx"): r for r in results if isinstance(r, dict)}

                for item in chunk:
                    idx = item["idx"]
                    text = item["text"]
                    result = by_idx.get(idx, {})
                    normalized = normalize_assignment_result(
                        result,
                        candidate_ids,
                        all_subtheme_ids,
                        max_codes,
                        low_thresh,
                    )
                    text_to_assignment[text].update(normalized)
                    # #region agent log
                    if "orange county" in (text or "").lower():
                        record_debug({
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "H28",
                            "location": "pipeline/assignment.py:assign_codes_two_stage:result",
                            "message": "Assignment result for target",
                            "data": {
                                "text": text,
                                "subtheme_ids": normalized.get("subtheme_ids", []),
                                "decision": normalized.get("decision"),
                                "confidence": normalized.get("confidence"),
                            },
                            "timestamp": int(time.time() * 1000),
                        })
                    # #endregion

            completed_groups += 1
            progress(40 + int((completed_groups / total_groups) * 50))
            status(f"Completed candidate group {completed_groups}/{total_groups}")

    all_assignments: List[Dict[str, Any]] = []
    for text, indices in text_to_indices.items():
        assignment = text_to_assignment.get(text)
        if not assignment:
            fallback_id = default_nonanswer_id or (all_subtheme_ids[0] if all_subtheme_ids else "T999.5")
            assignment = {
                "subtheme_ids": [fallback_id],
                "confidence": 0.0,
                "decision": "needs_review",
                "rationale": "Fallback: missing assignment",
            }

        subtheme_ids = assignment.get("subtheme_ids", [])
        confidence = assignment.get("confidence", 0.0)
        decision = assignment.get("decision", "needs_review")
        rationale = assignment.get("rationale", "")
        assignments_list = build_assignments_list(subtheme_ids, confidence) if subtheme_ids else []

        for idx in indices:
            all_assignments.append({
                "idx": idx,
                "assignments": assignments_list,
                "subtheme_ids": subtheme_ids,
                "confidence": confidence,
                "decision": decision,
                "rationale": rationale,
            })

    all_assignments.sort(key=lambda x: x["idx"])
    status("Assignment complete!")
    progress(100)
    flush_debug()
    return all_assignments, total_usage
