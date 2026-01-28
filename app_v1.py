"""
Express Explorer Thematic Coder — Streamlit MVP
Single‑file Streamlit app to upload survey open‑ends, auto‑discover Major/Sub themes,
assign single or multi‑codes with confidence, verify low‑confidence rows, and export XLSX.

Notes
- OpenAI API key is read from st.secrets["OPENAI_API_KEY"] or env var OPENAI_API_KEY.
- Uses Chat Completions with temperature 0 and optional seed for determinism.
- Always translates to English for coding, original text is preserved.
- Non‑answer handled as Major Theme = "Non‑answer" with Sub‑themes: Refusal, Don't know, Nonsense, Spam, Not applicable. "Other" sub-themes capture substantive outliers.
- Multi‑coding default with up to 3 codes. Single‑coding is a toggle.
- Dedupe strategy: theme discovery runs on unique texts with frequency weights.
  Assignment and charts honor full volume so output is one‑to‑one with input.
- Export format: <question>_thematic_coding_<YYYYMMDD>.xlsx with two sheets.

This is an MVP. The goal is correctness, clarity, and easy iteration.
"""

import os
import io
import json
import math
import datetime as dt
import re
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Tuple
from threading import Lock
import threading
import random
from collections import deque



import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from openai import RateLimitError
from dotenv import load_dotenv

from assignment_utils import (
    build_assignments_list,
    make_assignment_decision_schema,
    normalize_assignment_result,
    normalize_candidate_ids,
    stable_candidate_key,
)
from candidate_retrieval import (
    build_subtheme_records,
    embed_texts,
    get_candidate_ids,
    theme_signature,
)
from pipeline.assignment import (
    ASSIGNMENT_STAGE2_SYSTEM as PIPE_ASSIGNMENT_STAGE2_SYSTEM,
    ASSIGNMENT_STAGE2_USER_TEMPLATE as PIPE_ASSIGNMENT_STAGE2_USER_TEMPLATE,
    assign_codes_two_stage as pipeline_assign_codes_two_stage,
)
from pipeline.governance import govern_theme_dict as pipeline_govern_theme_dict
from pipeline.llm_client import (
    CacheStats,
    RateLimiter,
    SQLiteCache,
    compute_prompt_version,
    oai_json_completion as pipeline_oai_json_completion,
)
from pipeline.run_audit import build_run_audit, compute_run_id, dataset_signature
from pipeline.theme_discovery import (
    THEME_DISCOVERY_SYSTEM as PIPE_THEME_DISCOVERY_SYSTEM,
    THEME_DISCOVERY_USER as PIPE_THEME_DISCOVERY_USER,
    build_theme_frame_with_progress as pipeline_build_theme_frame_with_progress,
)
from pipeline.governance import (
    GOVERNANCE_SYSTEM as PIPE_GOVERNANCE_SYSTEM,
    GOVERNANCE_USER_TEMPLATE as PIPE_GOVERNANCE_USER_TEMPLATE,
)
from pipeline.utils import chunk_data, estimate_tokens, safe_prompt_token_budget
from theme_governance import (
    GOVERNANCE_SCHEMA,
    THEME_SCHEMA,
    apply_governance_change_log,
    normalize_change_log,
    normalize_theme_dict_order,
    validate_json_schema,
)


# ------------------------------
# Utilities
# ------------------------------

@st.cache_resource
def get_openai_client() -> OpenAI:
    """
    Prefer Streamlit secrets in deployed environments, fall back to env/.env locally.
    Cached so we don't re-create the client on every rerun.
    """
    api_key = None

    # Prefer Streamlit secrets (Streamlit Cloud / deployed)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    # Fall back to env/.env for local dev
    if not api_key:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OpenAI API key not set. Add OPENAI_API_KEY to Streamlit secrets or environment.")
        st.stop()

    # Add a sane timeout for reliability (supported by OpenAI Python 1.x)
    return OpenAI(
        api_key=api_key,
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "300")),
        max_retries=0,  # we handle retries explicitly (avoid double-retry) :contentReference[oaicite:11]{index=11}
    )


@st.cache_resource
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(DEFAULT_RPM, DEFAULT_TPM)


@st.cache_resource
def get_llm_cache() -> SQLiteCache:
    return SQLiteCache(os.path.join("cache", "llm_cache.sqlite"))


def get_cache_stats() -> CacheStats:
    if "_cache_stats" not in st.session_state:
        st.session_state["_cache_stats"] = CacheStats()
    return st.session_state["_cache_stats"]




def fmt_cost(total_prompt_tokens: int, total_completion_tokens: int, pricing: Dict[str, Any]) -> float:
    """Return estimated USD cost based on token usage and per‑1k pricing dict.
    pricing example: {"prompt_per_1k": 5e-6, "completion_per_1k": 1.5e-5}
    """
    p = pricing.get("prompt_per_1k", 0.0)
    c = pricing.get("completion_per_1k", 0.0)
    return (total_prompt_tokens / 1000.0) * p + (total_completion_tokens / 1000.0) * c


def today_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d")


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Normalize typical whitespace
    s = " ".join(s.split())
    return s


# Best-effort redaction for obvious PII before sending text to the LLM.
# This will NOT catch names/addresses reliably; advise users to pre-scrub if needed.
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
# Conservative US-centric phone matcher (reduces false positives vs ultra-generic digit patterns).
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def redact_pii(s: str) -> str:
    if not s:
        return ""
    s2 = s
    s2 = _EMAIL_RE.sub("[EMAIL]", s2)
    s2 = _URL_RE.sub("[URL]", s2)
    s2 = _PHONE_RE.sub("[PHONE]", s2)
    s2 = _SSN_RE.sub("[SSN]", s2)
    return s2



def is_empty_like(s: str) -> bool:
    if not s:
        return True
    # Keep conservative: "no/none/nothing" can be substantive depending on the question.
    na_set = {"n/a", "na", "-", "—", "dont know", "don't know", "idk", "prefer not to say"}
    return s.lower() in na_set



def estimate_tokens(text: str, model: str = "gpt-5") -> int:
    """
    Best-effort token estimation.
    - Uses tiktoken when available (more accurate)
    - Falls back to ~4 chars/token heuristic
    """
    try:
        import tiktoken  # optional dependency

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            # Fallback encoding used by many OpenAI models
            enc = tiktoken.get_encoding("o200k_base")

        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def safe_prompt_token_budget(model: str, reserve_output_tokens: int = 8_000) -> int:
    """
    Conservative prompt budget (tokens) for the request INPUT.
    GPT-5 family models have 400k context and 128k max output tokens. :contentReference[oaicite:5]{index=5}
    So max input is effectively bounded to ~272k (=400k-128k), and we keep extra safety headroom.
    """
    # Default conservative fallback if model is unknown
    context_window = 128_000
    max_output_tokens = 16_384

    if model.startswith("gpt-5"):
        context_window = 400_000
        max_output_tokens = 128_000  # per model docs :contentReference[oaicite:6]{index=6}

    max_input = max(1, context_window - max_output_tokens)
    # Keep safety headroom for: system text, JSON schema overhead, and tool formatting
    return max(1, int(max_input * 0.85) - reserve_output_tokens)


def chunk_data(data: list, max_tokens: int, model: str = "gpt-5") -> list:
    """
    Chunk a list of items so the JSON payload stays under max_tokens.
    Uses compact JSON encoding to reduce prompt size.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for item in data:
        item_str = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        item_tokens = estimate_tokens(item_str, model=model)

        # If a single item is huge, force it into its own chunk
        if current_chunk and (current_tokens + item_tokens > max_tokens):
            chunks.append(current_chunk)
            current_chunk = [item]
            current_tokens = item_tokens
        else:
            current_chunk.append(item)
            current_tokens += item_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ------------------------------
# Rate limiting (RPM + TPM)
# ------------------------------
DEFAULT_RPM = int(os.getenv("OPENAI_RPM", "500"))
DEFAULT_TPM = int(os.getenv("OPENAI_TPM", "500000"))

_rate_lock = Lock()
_req_times = deque()          # timestamps (seconds) for requests
_tok_times = deque()          # (timestamp, tokens) for token usage
_tok_sum = 0                  # rolling sum of tokens in last 60s


def check_rate_limits(estimated_tokens: int = 0, rpm: int = DEFAULT_RPM, tpm: int = DEFAULT_TPM) -> None:
    """
    Hard-enforced rolling 60s window limiter.
    Thread-safe, blocks until the request can be made.
    """
    global _tok_sum

    with _rate_lock:
        now = time.time()
        cutoff = now - 60.0

        # prune old requests
        while _req_times and _req_times[0] < cutoff:
            _req_times.popleft()

        # prune old tokens
        while _tok_times and _tok_times[0][0] < cutoff:
            ts, toks = _tok_times.popleft()
            _tok_sum -= toks

        # compute required sleep
        wait = 0.0

        if len(_req_times) >= rpm:
            wait = max(wait, (_req_times[0] + 60.0) - now)

        if estimated_tokens > 0 and (_tok_sum + estimated_tokens) > tpm:
            need_to_expire = (_tok_sum + estimated_tokens) - tpm
            running = 0
            for ts, toks in _tok_times:
                running += toks
                if running >= need_to_expire:
                    wait = max(wait, (ts + 60.0) - now)
                    break
            else:
                # extremely conservative fallback
                wait = max(wait, 1.0)

    if wait > 0:
        # sleep outside lock so other threads can compute their waits
        time.sleep(wait + random.uniform(0, 0.25))

    with _rate_lock:
        now = time.time()
        _req_times.append(now)
        if estimated_tokens > 0:
            _tok_times.append((now, estimated_tokens))
            _tok_sum += estimated_tokens


def _extract_status_code(err: Exception) -> int | None:
    # Best-effort: openai-python exceptions often expose status_code or a response with status_code
    for attr in ("status_code",):
        sc = getattr(err, attr, None)
        if isinstance(sc, int):
            return sc
    resp = getattr(err, "response", None)
    sc = getattr(resp, "status_code", None)
    return sc if isinstance(sc, int) else None


def _extract_retry_after_seconds(err: Exception) -> float | None:
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) or getattr(err, "headers", None) or {}
    if not isinstance(headers, dict):
        return None
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra is None:
        return None
    try:
        return float(ra)
    except Exception:
        return None


def retry_with_backoff(func, max_retries: int = 6, base_delay: float = 0.5, max_delay: float = 30.0):
    """
    Retries transient failures with exponential backoff + jitter.
    Keeps Streamlit calls out of worker threads.
    """
    transient_status = {408, 409, 425, 429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            status = _extract_status_code(e)
            is_transient = isinstance(e, RateLimitError) or (status in transient_status)

            # Heuristic fallback for network/timeout-y failures
            msg = str(e).lower()
            if ("timeout" in msg) or ("temporarily" in msg) or ("connection" in msg):
                is_transient = True

            if (not is_transient) or (attempt >= max_retries):
                raise

            retry_after = _extract_retry_after_seconds(e)
            if retry_after is not None:
                delay = min(max_delay, max(0.0, retry_after))
            else:
                delay = min(max_delay, base_delay * (2 ** attempt))

            # jitter
            delay = delay + random.uniform(0, delay * 0.2)
            time.sleep(delay)



def deduplicate_responses(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Deduplicate responses and return unique responses with their original indices"""
    unique_responses = {}
    response_to_indices = {}
    
    for row in rows:
        text = row["text"]
        idx = row["idx"]
        
        if text in unique_responses:
            # Add this index to the existing response
            response_to_indices[text].append(idx)
        else:
            # New unique response
            unique_responses[text] = row
            response_to_indices[text] = [idx]
    
    return list(unique_responses.values()), response_to_indices


def process_chunk_batch(client: OpenAI, model: str, theme_dict: Dict[str, Any], chunks: List[List[Dict[str, Any]]], max_codes: int, seed: int | None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Deprecated: use two-stage assignment pipeline."""
    raise NotImplementedError("process_chunk_batch is deprecated in favor of two-stage assignment.")


def process_chunk_batch_optimized(client: OpenAI, model: str, theme_dict: Dict[str, Any], chunks: List[List[Dict[str, Any]]], max_codes: int, seed: int | None, progress_bar=None, status_text=None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Deprecated: use two-stage assignment pipeline."""
    raise NotImplementedError("process_chunk_batch_optimized is deprecated in favor of two-stage assignment.")


# ------------------------------
# JSON Schemas for faster parsing
# ------------------------------

def allowed_subtheme_ids(theme_dict: Dict[str, Any]) -> List[str]:
    """Return all valid *subtheme* IDs (leaf codes) in stable order."""
    ids: List[str] = []
    for major in theme_dict.get("major_themes", []):
        for sub in (major.get("subs") or []):
            sid = sub.get("id")
            if sid:
                ids.append(sid)
    return sorted(set(ids))


def make_assignments_schema(allowed_ids: List[str], max_codes: int) -> Dict[str, Any]:
    """Dynamic JSON schema: constrain theme_id to enum(allowed_ids) and cap codes returned."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "assignments",
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "idx": {"type": "integer"},
                                "assignments": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": int(max_codes),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "theme_id": {"type": "string", "enum": allowed_ids},
                                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                        },
                                        "required": ["theme_id", "confidence"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["idx", "assignments"],
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

# ------------------------------
# Prompt builders
# ------------------------------


THEME_DISCOVERY_SYSTEM = (
    "You are a senior market research analyst. You design clear, business‑ready thematic taxonomies that provide actionable insights. Focus on the substantive content of what respondents are saying about the topic, not on survey mechanics or response quality. Use neutral, professional language. Specific themes are more valuable than generic 'Other' categories for business decision-making."
)

THEME_DISCOVERY_USER = (
    """
You will read a set of open‑ended responses for one survey question.
Create a hierarchical coding frame with Major Themes and Sub‑themes.
Goals:
- Capture the full variety by creating specific, meaningful Sub‑themes. Prioritize creating distinct Sub‑themes over generic "Other" categories.
- For each Major Theme and Sub-theme, include approx_pct ∈ [0,1] estimating coverage. Avoid Sub-themes below ~0.02 unless conceptually critical.
- Ensure Major Themes are at similar abstraction levels; avoid one ultra-broad Major vs. highly narrow peers.
- Keep Major Themes distinct and non‑overlapping.
- Each Sub‑theme should ladder under exactly one Major Theme.
- Create specific Sub‑themes even for smaller groups of similar responses (3+ responses with similar meaning warrant their own Sub‑theme).
- If an 'Other [Major Topic]' Sub-theme is unavoidable, cap approx_pct ≤ 0.05 and define it clearly. Never create a Major Theme named 'Other'.
- Provide a short definition for each theme.
- Example quotes ≤12 words; remove PII/URLs.
- Respect these non‑answer rules: Do not mix non‑answers with substantive themes. Use a separate Major Theme named "Non‑answer" with Sub‑themes among: Refusal, Don't know, Nonsense, Spam, Not applicable. Include only those that appear. When a non-answer pattern is common, name the Sub-theme to reflect the question's context (e.g., 'Unable to Name a Positive Association' rather than generic 'Don't know').
- Do NOT create themes about survey mechanics, selection processes, or respondent confusion unless responses explicitly mention problems with the survey itself. Focus on the substantive content of what respondents are saying.
- Use neutral, professional language. Avoid judgmental terms like "weak", "poor", "bad", or "invalid". Instead use descriptive terms like "brief", "general", or "unspecified".
- Assume responses are already translated to English.
- Consider the frequency weights when balancing the frame. Popular ideas should not be buried.

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
Return only valid JSON matching the schema; do not include any additional text, code fences, or commentary. All id and label values must be unique across the taxonomy.

IMPORTANT: Responses about preferences, desires, experiences, opinions, and reasons are SUBSTANTIVE CONTENT, not survey mechanics. Only classify responses as survey-related if they explicitly mention problems with the survey questions, confusion about instructions, or technical issues.
"""
)

ASSIGNMENT_SYSTEM = (
    "You are a meticulous qualitative coder. You assign responses to themes based on their substantive content, not on response quality or survey mechanics. Focus on what respondents are actually saying about the topic."
)

ASSIGNMENT_STAGE2_SYSTEM = (
    "You are a meticulous qualitative coder. You must choose from the provided candidate sub-themes only. "
    "Do not invent new themes."
)

ASSIGNMENT_USER_TEMPLATE = (
    """
You will assign the following responses to the provided theme dictionary.

For each response, assign up to {max_codes} distinct themes from the dictionary, each with a confidence score between 0 and 1. Only include multiple assignments when they capture materially different, valid facets of the response; otherwise prefer a single best theme. Order assignments by descending confidence.

Focus on the substantive content of what respondents are saying. Only assign to "Non-answer" themes if the response is truly a non-answer (refusal, don't know, nonsense, spam, or not applicable).

Return JSON in this exact format:
{{
  "results": [
    {{
      "idx": <row index integer>,
      "assignments": [
        {{"theme_id": "T1.2", "confidence": 0.87}}
      ]
    }}
  ]
}}

Theme dictionary:
{theme_json}

Responses:
{responses_json}
"""
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

Responses:
{responses_json}
"""
)

VERIFY_SYSTEM = (
    "You are a reviewer checking questionable assignments and fixing them."
)

VERIFY_USER_TEMPLATE = (
    """
You will re‑check only low‑confidence or ambiguous assignments. If an assignment is below {low_thresh}, try to improve it by selecting a better theme from the dictionary.

If the top assignment is <{low_thresh} and a secondary theme is a materially better fit, promote the secondary and adjust confidences accordingly.
If an assignment is clearly correct but under-confident, raise it to an appropriate level; do not leave obviously correct matches <{low_thresh}.
If no theme is defensible after re-check, keep a single, best-effort assignment with low confidence.


Return the same JSON shape as the assignment step, for only the provided rows. If you agree with the existing assignment, return it unchanged but you may adjust confidence.

Return only valid JSON matching the schema; do not include any additional text, code fences, or commentary.

Theme dictionary:
{theme_json}

Flagged rows:
{flagged_json}
"""
)

# Question label inference schema
QUESTION_LABEL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "question_label",
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"}
            },
            "required": ["label"],
            "additionalProperties": False
        },
        "strict": True
    }
}


def infer_question_label(client: OpenAI, model: str, question_text: str, seed: int | None) -> str:
    """Infer a compact question label like qAge or qGender from the question text."""
    if not question_text:
        return "qQuestion"

    # If already tagged like [qAge], prefer that tag
    m = re.match(r"^\s*\[([A-Za-z][A-Za-z0-9_]*)\]\s*", question_text)
    if m:
        tag = m.group(1)
        return f"q{tag[1:]}" if tag.lower().startswith("q") else f"q{tag}"

    system = "You assign concise variable-style question labels."
    user = (
        "Given this survey question text, propose a short label in the format qXxx. "
        "Use camel case, no spaces, only letters/numbers. Keep it under 12 chars if possible.\n\n"
        f"Question: {question_text}\n\n"
        "Return JSON: {\"label\":\"qXxx\"}"
    )

    data, _ = retry_with_backoff(
        lambda: oai_json_completion(
            client,
            model,
            system,
            user,
            seed,
            response_schema=QUESTION_LABEL_SCHEMA,
            reasoning_effort="minimal",
            verbosity="low",
            reserve_output_tokens=256,
        )
    )

    label = (data.get("label") if isinstance(data, dict) else None) or "qQuestion"
    label = re.sub(r"[^A-Za-z0-9]", "", label)
    if not label.lower().startswith("q"):
        label = f"q{label}"
    return label or "qQuestion"

# ------------------------------
# OpenAI helpers
# ------------------------------

class OAICounter:
    def __init__(self):
        self.prompt = 0
        self.completion = 0

    def add(self, usage: Dict[str, int]):
        if not usage:
            return
        self.prompt += usage.get("prompt_tokens", 0)
        self.completion += usage.get("completion_tokens", 0)



class JsonParseError(ValueError):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


def _schema_to_responses_text_format(response_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your schemas are currently in Chat Completions format:
      {"type":"json_schema","json_schema":{name,schema,strict}}
    Responses API expects:
      {"type":"json_schema","name":...,"schema":...,"strict":...}
    """
    if not response_schema:
        return {"type": "json_object"}

    if response_schema.get("type") == "json_schema" and "json_schema" in response_schema:
        js = response_schema["json_schema"]
        return {
            "type": "json_schema",
            "name": js.get("name", "schema"),
            "schema": js.get("schema", {}),
            "strict": bool(js.get("strict", True)),
        }

    # Already in the flattened shape
    return response_schema


def _should_use_responses_api(client: OpenAI, model: str) -> bool:
    # Prefer Responses for GPT-5 family (better support for reasoning/verbosity & migration path) :contentReference[oaicite:15]{index=15}
    return hasattr(client, "responses") and model.startswith("gpt-5")


def oai_json_completion(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    seed: int = 42,
    response_schema: Dict[str, Any] | None = None,
    reasoning_effort: str | None = "minimal",   # GPT-5: minimal/medium/high; GPT-5.1: none/low/...
    verbosity: str | None = "low",
    reserve_output_tokens: int = 8_000,
    return_raw: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, int]] | Tuple[Dict[str, Any], Dict[str, int], str]:
    limiter = get_rate_limiter()
    cache = get_llm_cache()
    cache_stats = get_cache_stats()
    data, usage, raw, _ = pipeline_oai_json_completion(
        client,
        model,
        system,
        user,
        seed,
        response_schema,
        limiter=limiter,
        cache=cache,
        cache_stats=cache_stats,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        reserve_output_tokens=reserve_output_tokens,
    )
    if return_raw:
        return data, usage, raw
    return data, usage


# ------------------------------
# JSON validation + repair
# ------------------------------

REPAIR_SYSTEM = "You are a JSON repair tool that returns strictly valid JSON."

REPAIR_USER_TEMPLATE = (
    "You will fix invalid JSON so it conforms exactly to the provided JSON schema.\n"
    "Return corrected JSON only. No commentary, no code fences.\n\n"
    "Invalid JSON:\n{invalid_json}\n\n"
    "JSON schema:\n{schema_json}\n"
)


def repair_json_to_schema(
    client: OpenAI,
    model: str,
    invalid_json: str,
    schema: Dict[str, Any],
    seed: int | None,
    context_label: str,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    user = REPAIR_USER_TEMPLATE.format(invalid_json=invalid_json, schema_json=schema_json)
    data, usage, _ = oai_json_completion(
        client,
        model,
        REPAIR_SYSTEM,
        user,
        seed,
        response_schema=schema,
        reasoning_effort="minimal",
        verbosity="low",
        reserve_output_tokens=4_000,
        return_raw=True,
    )
    ok, err = validate_json_schema(data, schema)
    if not ok:
        raise ValueError(f"{context_label}: repair failed schema validation: {err}")
    return data, usage


def validate_or_repair_json(
    client: OpenAI,
    model: str,
    parsed: Dict[str, Any],
    raw_text: str,
    schema: Dict[str, Any],
    seed: int | None,
    context_label: str,
) -> Tuple[Dict[str, Any], Dict[str, int], bool]:
    ok, err = validate_json_schema(parsed, schema)
    if ok:
        return parsed, {"prompt_tokens": 0, "completion_tokens": 0}, False

    invalid_text = raw_text or json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    repaired, usage = repair_json_to_schema(client, model, invalid_text, schema, seed, context_label)
    return repaired, usage, True


# ------------------------------
# Theming logic
# ------------------------------

def build_theme_frame(client: OpenAI, model: str, texts: List[str], freq: List[int], seed: int | None) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Deprecated: use build_theme_frame_with_progress for schema-safe discovery."""
    raise NotImplementedError(
        "build_theme_frame is deprecated and not safe for schema-validated discovery. "
        "Use build_theme_frame_with_progress instead."
    )


def calculate_dynamic_thresholds(theme_dict: Dict[str, Any], assigned_data: List[Dict] = None) -> Dict[str, float]:
    """Calculate dynamic thresholds based on data characteristics"""
    thresholds = {
        "tiny_threshold": 1.0,  # Default
        "low_confidence_threshold": 0.7,  # Default
        "other_usage_threshold": 10.0,  # Default
        "recommendations": []
    }
    
    if not theme_dict or not assigned_data:
        return thresholds
    
    # Analyze theme distribution
    theme_counts = {}
    total_assignments = 0
    
    for item in assigned_data:
        for assignment in item.get("assignments", []):
            theme_id = assignment.get("theme_id", "")
            if theme_id:
                theme_counts[theme_id] = theme_counts.get(theme_id, 0) + 1
                total_assignments += 1
    
    if total_assignments == 0:
        return thresholds
    
    # Calculate theme percentages
    theme_percentages = {theme_id: (count / total_assignments) * 100 for theme_id, count in theme_counts.items()}
    
    # Adjust tiny threshold based on theme distribution
    single_response_themes = sum(1 for pct in theme_percentages.values() if pct < 1.0)
    total_themes = len(theme_percentages)
    
    if total_themes > 0:
        single_response_ratio = single_response_themes / total_themes
        if single_response_ratio > 0.3:  # More than 30% are tiny themes
            thresholds["tiny_threshold"] = 0.5  # Lower threshold
            thresholds["recommendations"].append("Many single-response themes detected - consider consolidating")
        elif single_response_ratio < 0.1:  # Less than 10% are tiny themes
            thresholds["tiny_threshold"] = 2.0  # Higher threshold
            thresholds["recommendations"].append("Good theme distribution - can use higher tiny threshold")
    
    # Adjust confidence threshold based on assignment patterns
    confidence_scores = []
    for item in assigned_data:
        for assignment in item.get("assignments", []):
            conf = assignment.get("confidence", 0)
            if conf > 0:
                confidence_scores.append(conf)
    
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        if avg_confidence < 0.6:
            thresholds["low_confidence_threshold"] = 0.5  # Lower threshold
            thresholds["recommendations"].append("Low average confidence - consider lowering review threshold")
        elif avg_confidence > 0.8:
            thresholds["low_confidence_threshold"] = 0.8  # Higher threshold
            thresholds["recommendations"].append("High average confidence - can use higher review threshold")
    
    # Check "Other" theme usage
    other_themes = [theme_id for theme_id in theme_percentages.keys() if "other" in theme_id.lower()]
    if other_themes:
        other_usage = sum(theme_percentages[theme_id] for theme_id in other_themes)
        if other_usage > 20:
            thresholds["other_usage_threshold"] = 15.0  # Lower threshold
            thresholds["recommendations"].append("High 'Other' usage - consider creating more specific themes")
        elif other_usage < 5:
            thresholds["other_usage_threshold"] = 15.0  # Higher threshold
            thresholds["recommendations"].append("Low 'Other' usage - good theme specificity")
    
    return thresholds

def analyze_low_confidence_patterns(low_confidence_responses: List[Dict], question_context: Dict = None) -> Dict[str, Any]:
    """Analyze low-confidence responses to identify potential new theme patterns"""
    if not low_confidence_responses:
        return {"patterns": [], "suggestions": []}
    
    # Group responses by their current assignments to find patterns
    assignment_groups = {}
    for response in low_confidence_responses:
        assignments = response.get("assignments", [])
        if assignments:
            # Group by the primary theme they were assigned to
            primary_theme = assignments[0].get("theme_id", "unknown")
            if primary_theme not in assignment_groups:
                assignment_groups[primary_theme] = []
            assignment_groups[primary_theme].append(response)
    
    patterns = []
    suggestions = []
    
    # Analyze each group for potential new themes
    for theme_id, responses in assignment_groups.items():
        if len(responses) >= 3:  # Need at least 3 responses to suggest a pattern
            response_texts = [r.get("text", "") for r in responses]
            
            # Look for common patterns in the text
            common_words = {}
            for text in response_texts:
                words = text.lower().split()
                for word in words:
                    if len(word) > 3:  # Only consider meaningful words
                        common_words[word] = common_words.get(word, 0) + 1
            
            # Find words that appear in multiple responses
            frequent_words = {word: count for word, count in common_words.items() if count >= 2}
            
            if frequent_words:
                # Suggest a new sub-theme based on the pattern
                most_common = max(frequent_words.items(), key=lambda x: x[1])
                pattern_name = f"New Pattern: {most_common[0].title()}"
                
                patterns.append({
                    "theme_id": theme_id,
                    "pattern_name": pattern_name,
                    "response_count": len(responses),
                    "common_words": list(frequent_words.keys())[:5],
                    "sample_responses": response_texts[:3]
                })
                
                suggestions.append(f"Consider adding sub-theme '{most_common[0].title()}' under theme {theme_id}")
    
    return {
        "patterns": patterns,
        "suggestions": suggestions,
        "total_low_confidence": len(low_confidence_responses)
    }

def add_new_theme_to_dictionary(theme_dict: Dict[str, Any], new_theme: Dict[str, Any]) -> Dict[str, Any]:
    """Add a new theme or sub-theme to the existing theme dictionary"""
    updated_dict = theme_dict.copy()
    major_themes = updated_dict.get("major_themes", [])
    
    if new_theme.get("type") == "major_theme":
        # Add new major theme
        # Find the next available theme ID
        max_id = 0
        for major in major_themes:
            theme_id = major.get("id", "")
            if theme_id.startswith("T") and theme_id[1:].isdigit():
                max_id = max(max_id, int(theme_id[1:]))
        
        new_major_id = f"T{max_id + 1}"
        new_major_theme = {
            "id": new_major_id,
            "label": new_theme.get("theme_name", "New Theme"),
            "definition": new_theme.get("definition", "New theme definition"),
            "approx_pct": 0.05,  # Default small percentage
            "subs": []
        }
        major_themes.append(new_major_theme)
        
    elif new_theme.get("type") == "sub_theme":
        # Add new sub-theme to existing major theme
        parent_theme_id = new_theme.get("parent_theme_id", "")
        
        for major in major_themes:
            if major.get("id") == parent_theme_id:
                subs = major.get("subs", [])
                
                # Find next sub-theme ID
                max_sub_id = 0
                for sub in subs:
                    sub_id = sub.get("id", "")
                    if "." in sub_id:
                        try:
                            sub_num = int(sub_id.split(".")[1])
                            max_sub_id = max(max_sub_id, sub_num)
                        except:
                            pass
                
                new_sub_id = f"{parent_theme_id}.{max_sub_id + 1}"
                new_sub_theme = {
                    "id": new_sub_id,
                    "label": new_theme.get("theme_name", "New Sub-theme"),
                    "definition": new_theme.get("definition", "New sub-theme definition"),
                    "approx_pct": 0.02,  # Default small percentage
                    "examples": new_theme.get("sample_responses", [])[:3]
                }
                subs.append(new_sub_theme)
                break
    
    updated_dict["major_themes"] = major_themes
    return updated_dict

def suggest_new_themes_from_review(low_confidence_responses: List[Dict], existing_theme_dict: Dict, question_context: Dict = None) -> Dict[str, Any]:
    """Use AI to suggest new themes based on low-confidence responses"""
    if not low_confidence_responses or len(low_confidence_responses) < 3:
        return {"suggestions": [], "reasoning": "Not enough low-confidence responses to analyze"}
    
    # Prepare sample responses for AI analysis
    sample_responses = [r.get("text", "") for r in low_confidence_responses[:10]]  # Limit to 10 for efficiency
    
    # Create a focused prompt for theme suggestion
    suggestion_prompt = f"""
    You are analyzing low-confidence theme assignments to suggest new themes or sub-themes.
    
    Current theme dictionary structure:
    {json.dumps(existing_theme_dict, indent=2)[:1000]}...
    
    Low-confidence responses that don't fit well:
    {json.dumps(sample_responses, indent=2)}
    
    Question context: {question_context.get('focus', 'General analysis') if question_context else 'General analysis'}
    
    Analyze these responses and suggest:
    1. New sub-themes that could be added to existing major themes
    2. New major themes if the responses represent a completely different category
    3. Specific theme names and definitions
    
    Return JSON in this format:
    {{
      "suggestions": [
        {{
          "type": "sub_theme" or "major_theme",
          "parent_theme_id": "T1" (for sub-themes only),
          "theme_name": "Suggested Theme Name",
          "definition": "One sentence definition",
          "reasoning": "Why this theme is needed",
          "sample_responses": ["response1", "response2"]
        }}
      ]
    }}
    """
    
    return {"suggestions": [], "reasoning": "AI analysis not implemented yet"}

def validate_theme_quality(theme_dict: Dict[str, Any], question_context: Dict = None) -> Dict[str, Any]:
    """Validate theme quality and provide improvement suggestions"""
    validation_results = {
        "overall_score": 0.0,
        "issues": [],
        "suggestions": [],
        "theme_analysis": {}
    }
    
    major_themes = theme_dict.get("major_themes", [])
    total_score = 0.0
    
    for major in major_themes:
        major_id = major.get("id", "")
        major_label = major.get("label", "")
        subs = major.get("subs", [])
        
        theme_score = 0.0
        theme_issues = []
        theme_suggestions = []
        
        # Check theme balance
        if len(subs) < 2:
            theme_issues.append("Only one sub-theme - consider adding more granular categories")
            theme_score -= 0.2
        elif len(subs) > 8:
            theme_issues.append("Too many sub-themes - consider consolidating similar ones")
            theme_score -= 0.1
        
        # Check for "Other" overuse
        other_subs = [sub for sub in subs if "other" in sub.get("label", "").lower()]
        if len(other_subs) > 1:
            theme_issues.append("Multiple 'Other' sub-themes - consider consolidating")
            theme_score -= 0.3
        elif len(other_subs) == 1 and len(subs) > 3:
            other_pct = other_subs[0].get("approx_pct", 0)
            if other_pct > 0.15:
                theme_issues.append("'Other' sub-theme too large - consider creating specific themes")
                theme_score -= 0.2
        
        # Check theme specificity
        if "other" in major_label.lower():
            theme_issues.append("Major theme named 'Other' - should be more specific")
            theme_score -= 0.4
        
        # Check coverage estimates
        total_coverage = sum(sub.get("approx_pct", 0) for sub in subs)
        if total_coverage > 1.1:
            theme_issues.append("Coverage estimates exceed 100% - review approx_pct values")
            theme_score -= 0.1
        elif total_coverage < 0.7:
            theme_issues.append("Low coverage estimates - may be missing themes")
            theme_score -= 0.1
        
        # Question context alignment
        if question_context and question_context.get("type") != "general":
            priority_themes = question_context.get("priority_themes", [])
            if not any(priority in major_label.lower() for priority in [p.lower() for p in priority_themes]):
                theme_suggestions.append(f"Consider aligning with priority theme areas: {', '.join(priority_themes)}")
        
        # Calculate final theme score
        theme_score = max(0.0, min(1.0, theme_score + 0.8))  # Base score of 0.8
        
        validation_results["theme_analysis"][major_id] = {
            "label": major_label,
            "score": theme_score,
            "issues": theme_issues,
            "suggestions": theme_suggestions
        }
        
        total_score += theme_score
        validation_results["issues"].extend(theme_issues)
        validation_results["suggestions"].extend(theme_suggestions)
    
    # Calculate overall score
    if major_themes:
        validation_results["overall_score"] = total_score / len(major_themes)
    else:
        validation_results["overall_score"] = 0.0
        validation_results["issues"].append("No themes generated")
    
    # Overall suggestions
    if validation_results["overall_score"] < 0.7:
        validation_results["suggestions"].append("Consider regenerating themes with more specific focus")
    
    if len(validation_results["issues"]) == 0:
        validation_results["suggestions"].append("Theme quality looks good! Consider fine-tuning based on assignment results.")

    return validation_results


def ensure_nonanswer_theme(theme_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a Non-answer major theme exists with predictable leaf subthemes.
    Prevents brittle fallbacks and makes non-answer handling auditable.
    """
    majors = theme_dict.get("major_themes", []) or []

    def norm(x: str) -> str:
        return re.sub(r"[^a-z]+", "", (x or "").lower())

    # Locate existing Non-answer major if present
    nonanswer_major = None
    for m in majors:
        if norm(m.get("label", "")) == "nonanswer":
            nonanswer_major = m
            break

    # Create it if missing
    if not nonanswer_major:
        nonanswer_major = {
            "id": "T999",
            "label": "Non-answer",
            "definition": "Responses that do not provide a substantive answer to the question.",
            "approx_pct": 0.0,
            "subs": [],
        }
        majors.append(nonanswer_major)
        theme_dict["major_themes"] = majors

    # Ensure the Non-answer major has a stable id
    if not nonanswer_major.get("id"):
        nonanswer_major["id"] = "T999"

    major_id = nonanswer_major.get("id") or "T999"
    nonanswer_major.setdefault("subs", [])
    subs = nonanswer_major.get("subs", []) or []

    # Canonical Non-answer subthemes (match by normalized label)
    canonical = [
        ("refusal", "Refusal", "Explicit refusal to answer."),
        ("dontknow", "Don't know", "Respondent indicates they don't know / can't recall."),
        ("nonsense", "Nonsense", "Incoherent or meaningless text."),
        ("spam", "Spam", "Promotional/irrelevant content, links, or spam."),
        ("notapplicable", "Not applicable", "Response indicates question does not apply / blank / n/a."),
    ]

    existing = {norm(s.get("label", "")): s for s in subs}

    # Track used numeric suffixes for ids like <major_id>.<n>
    used_nums = set()
    for s in subs:
        sid = str(s.get("id", ""))
        m = re.match(rf"^{re.escape(major_id)}\.(\d+)$", sid)
        if m:
            used_nums.add(int(m.group(1)))

    next_n = 1
    for key, label, definition in canonical:
        if key in existing:
            # Ensure required fields exist
            existing[key].setdefault("definition", definition)
            existing[key].setdefault("examples", [])
            existing[key].setdefault("approx_pct", 0.0)
            continue

        while next_n in used_nums:
            next_n += 1

        subs.append({
            "id": f"{major_id}.{next_n}",
            "label": label,
            "definition": definition,
            "approx_pct": 0.0,
            "examples": [],
        })
        used_nums.add(next_n)
        next_n += 1

    nonanswer_major["subs"] = subs
    return theme_dict


def calibrate_confidence(confidence: float, response_text: str, theme_id: str) -> float:
    """Calibrate confidence based on response characteristics and theme fit"""
    calibrated = confidence
    
    # Response length factor (longer responses often more confident)
    response_length = len(response_text.split())
    if response_length < 5:
        calibrated *= 0.8  # Reduce confidence for very short responses
    elif response_length > 20:
        calibrated *= 1.1  # Boost confidence for detailed responses
        calibrated = min(calibrated, 1.0)  # Cap at 1.0
    
    # Theme specificity factor
    if "other" in theme_id.lower():
        calibrated *= 0.9  # Slightly reduce confidence for "Other" themes
    
    # Non-answer factor
    if "non-answer" in theme_id.lower():
        calibrated *= 0.85  # Reduce confidence for non-answer assignments
    
    # Ensure confidence stays within bounds
    return max(0.0, min(1.0, calibrated))

def detect_question_type(question_text):
    """Detect question type and return optimized theme discovery parameters"""
    question_lower = question_text.lower()
    
    if any(phrase in question_lower for phrase in ["worth the cost", "value", "worth it", "justify"]):
        return {
            "type": "cost_value",
            "focus": "Focus on cost-benefit analysis, value propositions, and economic reasoning",
            "priority_themes": ["Cost Concerns", "Value Proposition", "Budget Priorities", "ROI Analysis"]
        }
    elif any(phrase in question_lower for phrase in ["why", "reason", "motivation"]):
        return {
            "type": "reasoning",
            "focus": "Focus on underlying motivations, causal relationships, and explanatory factors",
            "priority_themes": ["Motivations", "Causal Factors", "Underlying Reasons", "Drivers"]
        }
    elif any(phrase in question_lower for phrase in ["prefer", "like", "favorite", "choose"]):
        return {
            "type": "preference",
            "focus": "Focus on preferences, choices, and comparative evaluations",
            "priority_themes": ["Preferences", "Choices", "Comparisons", "Rankings"]
        }
    elif any(phrase in question_lower for phrase in ["experience", "feel", "think", "opinion"]):
        return {
            "type": "experience",
            "focus": "Focus on experiences, emotions, perceptions, and subjective evaluations",
            "priority_themes": ["Experiences", "Emotions", "Perceptions", "Subjective Views"]
        }
    else:
        return {
            "type": "general",
            "focus": "Focus on comprehensive thematic analysis covering all aspects",
            "priority_themes": ["General Themes", "Diverse Responses", "Comprehensive Coverage"]
        }

def build_theme_frame_with_progress(client: OpenAI, model: str, texts: List[str], freq: List[int], seed: int | None, progress_bar, status_text, question_context: Dict = None) -> Tuple[Dict[str, Any], Dict[str, int]]:
    limiter = get_rate_limiter()
    cache = get_llm_cache()
    cache_stats = get_cache_stats()
    # #region agent log
    try:
        guidance_text = st.session_state.get(f"question_guidance::{text_col}", "").strip() if "text_col" in globals() else ""
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H6",
                "location": "app_v1.py:build_theme_frame_with_progress:wrapper",
                "message": "Calling pipeline theme discovery",
                "data": {
                    "guidance_len": len(guidance_text),
                    "text_count": len(texts or []),
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        
    # #endregion
    # #region agent log
    try:
        import sys
        td_mod = sys.modules.get("pipeline.theme_discovery")
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H15",
                "location": "app_v1.py:build_theme_frame_with_progress:module",
                "message": "Theme discovery module info",
                "data": {
                    "module_file": getattr(td_mod, "__file__", None),
                    "module_loaded": td_mod is not None,
                },
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        
    # #endregion

    return pipeline_build_theme_frame_with_progress(
        client,
        model,
        texts,
        freq,
        seed,
        limiter=limiter,
        cache=cache,
        cache_stats=cache_stats,
        question_context=question_context,
        guidance_text=st.session_state.get(f"question_guidance::{text_col}", "").strip() if "text_col" in globals() else "",
        normalize_locations=bool(st.session_state.get("normalize_locations", False)),
        geocode_user_agent=st.session_state.get("nominatim_user_agent", "").strip(),
        guidance_target_type=st.session_state.get("guidance_target_type", "auto"),
        guidance_soft_prefer=True,
        on_status=status_text.text if status_text else None,
        on_progress=progress_bar.progress if progress_bar else None,
        on_notice=st.info,
    )



def merge_theme_chunks(theme_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge theme chunks and deduplicate similar themes with proper hierarchy enforcement.

    Chunked theme discovery often produces colliding IDs (e.g., each chunk starts at T1/T1.1).
    This merge is label-based and then renumbers IDs deterministically to avoid overwrites and
    improve rerun consistency.
    """
    if not theme_chunks:
        return []

    def norm_label(x: str) -> str:
        return re.sub(r"\s+", " ", (x or "").strip().lower())

    def norm_alpha(x: str) -> str:
        return re.sub(r"[^a-z]+", "", (x or "").lower())

    # Step 1: Detect labels that appear as both major and sub across chunks (hierarchy conflicts)
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
    if conflicting_labels:
        st.warning(
            f"⚠️ Found {len(conflicting_labels)} themes that appear as both major and sub themes across chunks. "
            "These will be kept as major themes and removed from sub-themes."
        )

    # Step 2: Merge majors by normalized label (NOT by id)
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

        # Merge subthemes (unique by normalized label)
        existing_sub_keys = {norm_label(s.get("label", "")): s for s in merged_major.get("subs", []) if s.get("label")}

        for sub in theme.get("subs", []) or []:
            sub_label = (sub.get("label") or "").strip()
            if not sub_label:
                continue

            sub_key = norm_label(sub_label)

            # Remove hierarchy conflicts: if it is also a major label, keep it only at major level
            if sub_key in conflicting_labels:
                continue

            if sub_key in existing_sub_keys:
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

    # Step 3: Ensure no sub theme has the same label as its parent major theme
    for theme in merged_themes:
        major_label_norm = norm_label(theme.get("label", ""))
        theme["subs"] = [
            sub for sub in theme.get("subs", [])
            if norm_label(sub.get("label", "")) != major_label_norm
        ]

    # Step 4: Deterministic ordering + renumbering to avoid chunk ID collisions
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

    if conflicting_labels:
        st.success(f"✅ Resolved {len(conflicting_labels)} hierarchy conflicts and renumbered IDs for consistency.")

    return ordered


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
{{
  "action": "merge|split|rename|move",
  "from_ids": ["..."],
  "to_ids": ["..."],
  "reason": "short explanation"
}}

Question context (if available):
{question_context}

Theme dictionary:
{theme_json}
"""
)


def govern_theme_dict(
    client: OpenAI,
    model: str,
    theme_dict: Dict[str, Any],
    question_context: Dict[str, Any] | None,
    seed: int | None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
    limiter = get_rate_limiter()
    cache = get_llm_cache()
    cache_stats = get_cache_stats()
    return pipeline_govern_theme_dict(
        client,
        model,
        theme_dict,
        question_context,
        seed,
        limiter=limiter,
        cache=cache,
        cache_stats=cache_stats,
        on_notice=st.warning,
    )


def assign_codes(client: OpenAI, model: str, theme_dict: Dict[str, Any], rows: List[Dict[str, Any]], max_codes: int, seed: int | None,
                 low_thresh: float, top_k: int, embedding_model: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    return assign_codes_two_stage(
        client,
        model,
        theme_dict,
        rows,
        max_codes,
        seed,
        low_thresh,
        top_k,
        embedding_model,
        progress_bar=None,
        status_text=None,
    )
    
    # Step 1: Aggressive pre-filtering and deduplication
    
    # Pre-filter obvious non-answers and very short responses, but keep track of filtered ones
    filtered_rows = []
    non_answer_assignments = []
    
    # Find a non-answer theme ID from the theme dictionary
    non_answer_theme_id = None
    for major in theme_dict.get("major_themes", []):
        if "non-answer" in major.get("label", "").lower():
            # Use the first sub-theme of the non-answer major theme
            subs = major.get("subs", [])
            if subs:
                non_answer_theme_id = subs[0].get("id")
                break
    
    # Fallback if no non-answer theme found
    if not non_answer_theme_id:
        # Use the first available theme as fallback
        for major in theme_dict.get("major_themes", []):
            subs = major.get("subs", [])
            if subs:
                non_answer_theme_id = subs[0].get("id")
                break
    
    for row in rows:
        text = row["text"]
        if not text or is_empty_like(text) or len(text.strip()) < 5:
            # Assign to non-answer theme for filtered responses
            non_answer_assignments.append({
                "idx": row["idx"],
                "assignments": [{"theme_id": (non_answer_theme_id or "T99.1"), "confidence": 1.0}]
            })
            continue
        filtered_rows.append(row)
    
    # Deduplicate responses
    unique_rows, response_to_indices = deduplicate_responses(filtered_rows)
    
    original_count = len(rows)
    filtered_count = len(filtered_rows)
    unique_count = len(unique_rows)
    
    
    
    # Disabled fast assignment for maximum quality - all responses go to GPT-5
    # if unique_count > 20:  # Lower threshold for better performance
    #     st.info("⚡ Using fast assignment mode for common responses...")
    #     unique_rows = fast_assign_common_responses(unique_rows, theme_dict, max_codes)
    #     
    #     # Split already-assigned from raw rows (CRITICAL FIX)
    #     fast_done = [r for r in unique_rows if "assignments" in r]   # already assigned
    #     to_model = [r for r in unique_rows if "assignments" not in r]  # needs AI processing
    
    # Step 2: Smart batching with larger chunks - all responses go to GPT-5
    # CRITICAL FIX: Re-index unique rows with sequential indices for AI processing
    unique_rows_for_ai = []
    for i, row in enumerate(unique_rows):
        ai_row = row.copy()
        ai_row["idx"] = i  # Use sequential index for AI processing
        unique_rows_for_ai.append(ai_row)
    
    total_tokens = estimate_tokens(json.dumps(unique_rows_for_ai))
    
    if total_tokens <= 500000:  # GPT-5 safe token limit
        # Single request for unique responses
        
        theme_json = json.dumps(slim_theme_for_assignment(theme_dict))
        allowed_ids = allowed_subtheme_ids(theme_dict)
        schema = make_assignments_schema(allowed_ids, max_codes if allow_multicode else 1)

        responses_json = json.dumps(unique_rows_for_ai, separators=(",", ":"))
        
        # Debug: Show theme structure
        
        
        user = ASSIGNMENT_USER_TEMPLATE.format(max_codes=max_codes, theme_json=theme_json, responses_json=responses_json)
        
        def make_request():
            return oai_json_completion(client, model, ASSIGNMENT_SYSTEM, user, seed, schema)

        data, usage = retry_with_backoff(make_request)
        # Data expected as object with results array
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        else:
            # Fallback for backward compatibility
            data = data if isinstance(data, list) else []
        
        # All responses processed by GPT-5 - no fast assignment merging needed
        final_unique = data
        
        # Expand results back to all original responses
        expanded_assignments = expand_deduplicated_results(final_unique, response_to_indices)
        
        
        # Add back the non-answer assignments for filtered responses
        all_assignments = expanded_assignments + non_answer_assignments
        all_assignments.sort(key=lambda x: x["idx"])  # Sort by original index
        
        return all_assignments, usage
    
    else:
        # Process with larger chunks and more aggressive parallel processing
        chunks = chunk_data(unique_rows_for_ai, max_tokens=350000)  # Conservative chunks for GPT-5
        pass
        
        # Use conservative chunks for GPT-5 quality
        
        # Process chunks with more parallelism
        all_assignments, total_usage = process_chunk_batch_optimized(client, model, theme_dict, chunks, max_codes, seed)
        
        # Expand results back to all original responses
        expanded_assignments = expand_deduplicated_results(all_assignments, response_to_indices)
        
        
        # Add back the non-answer assignments for filtered responses
        final_assignments = expanded_assignments + non_answer_assignments
        final_assignments.sort(key=lambda x: x["idx"])  # Sort by original index
        
        pass
        return final_assignments, total_usage


def assign_codes_with_progress(client: OpenAI, model: str, theme_dict: Dict[str, Any], rows: List[Dict[str, Any]],
                               max_codes: int, seed: int | None, low_thresh: float, top_k: int, embedding_model: str,
                               progress_bar, status_text) -> Tuple[
    List[Dict[str, Any]], Dict[str, int]]:
    return assign_codes_two_stage(
        client,
        model,
        theme_dict,
        rows,
        max_codes,
        seed,
        low_thresh,
        top_k,
        embedding_model,
        progress_bar=progress_bar,
        status_text=status_text,
    )
    """Assign themes to responses with progress tracking.

    Improvements vs v1:
    - Ensures Non-answer leaf codes exist and uses simple heuristics to route empty/NA/don't-know/refusal correctly.
    - Uses token-budget batching (vs fixed chunk_size=10) to reduce repeated codebook overhead and API calls.
    - Propagates a lightweight rationale for auto/fallback paths to aid audit/review.
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    status_text.text("Processing responses...")
    progress_bar.progress(5)

    # Always ensure predictable Non-answer subthemes exist (even for uploaded codebooks)
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

    # Step 1: Build mapping of unique responses to their indices (to maintain consistency for duplicates)
    status_text.text("Deduplicating responses...")
    progress_bar.progress(10)

    text_to_indices: Dict[str, List[int]] = {}
    for row in rows:
        text = row.get("text", "").strip()
        idx = row.get("idx")
        if idx is None:
            continue

        if text not in text_to_indices:
            text_to_indices[text] = []
        text_to_indices[text].append(idx)

    unique_texts = list(text_to_indices.keys())

    # Prepare a compact theme representation once (reused across batches)
    theme_json = json.dumps(slim_theme_for_assignment(theme_dict), separators=(",", ":"))
    theme_tokens = estimate_tokens(theme_json)

    # Token-budget batching to reduce API calls (conservative default ~128k context models)
    max_prompt_tokens = 120000
    overhead_tokens = 1200  # system/user instructions, schema wrapper, etc.
    batch_budget = max(20000, max_prompt_tokens - theme_tokens - overhead_tokens)

    all_chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for text in unique_texts:
        # Rough per-item cost: payload framing + the text itself
        item_tokens = estimate_tokens(text) + 20
        if current_chunk and (current_tokens + item_tokens) > batch_budget:
            all_chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(text)
        current_tokens += item_tokens

    if current_chunk:
        all_chunks.append(current_chunk)

    # Step 2: Process chunks in parallel
    status_text.text(f"Processing {len(unique_texts)} unique responses in {len(all_chunks)} batches...")
    progress_bar.progress(20)

    _allowed_ids = allowed_subtheme_ids(theme_dict)
    schema = make_assignments_schema(_allowed_ids, max_codes=max_codes)

    # fallback used only if an LLM call fails or returns incomplete results
    fallback_theme_id = default_nonanswer_id or (_allowed_ids[0] if _allowed_ids else "T999.5")

    # Store assignments by text
    text_to_assignment: Dict[str, Dict[str, Any]] = {}

    def process_single_chunk(chunk_texts: List[str], chunk_idx: int) -> Dict[str, Dict[str, Any]]:
        chunk_assignments: Dict[str, Dict[str, Any]] = {}

        # Handle empty/non-answer responses deterministically without an LLM call
        responses_for_ai: List[Dict[str, Any]] = []
        for text in chunk_texts:
            if not text or is_empty_like(text):
                chunk_assignments[text] = {
                    "assignments": [{"theme_id": _pick_nonanswer_id(text), "confidence": 1.0}],
                    "rationale": "Auto: non-answer/blank",
                }
            else:
                responses_for_ai.append({"idx": len(responses_for_ai), "text": text})

        # If everything in this batch was non-answer, we're done
        if not responses_for_ai:
            return chunk_assignments

        responses_json = json.dumps(responses_for_ai, separators=(",", ":"))
        user_prompt = ASSIGNMENT_USER_TEMPLATE.format(theme_json=theme_json, responses_json=responses_json,
                                                      max_codes=max_codes)

        try:
            data, usage = retry_with_backoff(
                lambda: oai_json_completion(client, model, ASSIGNMENT_SYSTEM, user_prompt, seed, response_schema=schema)
            )

            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

            ai_assignments = data.get("results") if isinstance(data, dict) else data
            if not isinstance(ai_assignments, list):
                ai_assignments = []

            for assignment in ai_assignments:
                ai_idx = assignment.get("idx")
                if ai_idx is None or ai_idx < 0 or ai_idx >= len(responses_for_ai):
                    continue

                original_text = responses_for_ai[ai_idx]["text"]
                assignments = assignment.get("assignments", [])

                if assignments:
                    # Calibrate confidence for better quality
                    for a in assignments:
                        if "confidence" in a:
                            a["confidence"] = calibrate_confidence(a["confidence"], original_text,
                                                                   a.get("theme_id", ""))

                    chunk_assignments[original_text] = {"assignments": assignments}

            # Fill missing items (should be rare) with a low-confidence fallback for manual review
            for r in responses_for_ai:
                t = r["text"]
                if t not in chunk_assignments:
                    chunk_assignments[t] = {
                        "assignments": [{"theme_id": fallback_theme_id, "confidence": 0.0}],
                        "rationale": "Fallback: missing LLM result",
                    }

        except Exception as e:
            # On error, keep output shape stable but force manual review via 0-confidence
            for r in responses_for_ai:
                t = r["text"]
                chunk_assignments[t] = {
                    "assignments": [{"theme_id": fallback_theme_id, "confidence": 0.0}],
                    "rationale": f"Fallback: LLM error ({type(e).__name__})",
                }

        return chunk_assignments

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(all_chunks))) as executor:
        future_to_chunk = {executor.submit(process_single_chunk, chunk, i): i for i, chunk in enumerate(all_chunks)}

        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_assignments = future.result()
                text_to_assignment.update(chunk_assignments)
                completed += 1

                progress_pct = 20 + int((completed / len(all_chunks)) * 70)
                progress_bar.progress(progress_pct)
                status_text.text(f"Completed batch {completed}/{len(all_chunks)}")

            except Exception as e:
                st.error(f"Error processing batch {chunk_idx}: {str(e)}")

    # Step 3: Expand assignments to all original responses (maintaining duplicates)
    all_assignments: List[Dict[str, Any]] = []

    for text, indices in text_to_indices.items():
        assignment_template = text_to_assignment.get(text)
        if assignment_template:
            for idx in indices:
                all_assignments.append({
                    "idx": idx,
                    "assignments": assignment_template.get("assignments", []),
                    "rationale": assignment_template.get("rationale", ""),
                })
        else:
            for idx in indices:
                all_assignments.append({
                    "idx": idx,
                    "assignments": [{"theme_id": fallback_theme_id, "confidence": 0.0}],
                    "rationale": "Fallback: missing assignment",
                })

    # Step 4: Ensure we have assignments for ALL response indices
    assigned_indices = {a["idx"] for a in all_assignments}
    total_rows = len(rows)

    for i in range(total_rows):
        if i not in assigned_indices:
            all_assignments.append({
                "idx": i,
                "assignments": [{"theme_id": fallback_theme_id, "confidence": 0.0}],
                "rationale": "Fallback: missing response index",
            })

    all_assignments.sort(key=lambda x: x["idx"])

    progress_bar.progress(100)
    status_text.text(f"Completed processing {len(all_assignments)} assignments")

    return all_assignments, total_usage


def fast_assign_common_responses(unique_rows: List[Dict[str, Any]], theme_dict: Dict[str, Any], max_codes: int) -> List[Dict[str, Any]]:
    """Fast assignment for very common responses using pattern matching"""
    
    # Build a mapping of actual theme IDs to their labels for better matching
    theme_mapping = {}
    for major in theme_dict.get("major_themes", []):
        theme_mapping[major["id"]] = major["label"].lower()
        for sub in major.get("subs", []):
            theme_mapping[sub["id"]] = sub["label"].lower()
    
    # Find themes that match common patterns
    positive_themes = []
    negative_themes = []
    neutral_themes = []
    non_answer_themes = []
    
    for theme_id, label in theme_mapping.items():
        if any(word in label for word in ["good", "positive", "satisfied", "happy", "love", "excellent", "great", "amazing", "perfect", "wonderful", "fantastic"]):
            positive_themes.append(theme_id)
        elif any(word in label for word in ["bad", "negative", "dissatisfied", "unhappy", "hate", "terrible", "awful", "worst", "horrible", "disappointed", "poor"]):
            negative_themes.append(theme_id)
        elif any(word in label for word in ["okay", "ok", "fine", "average", "decent", "alright", "neutral", "moderate"]):
            neutral_themes.append(theme_id)
        elif any(word in label for word in ["n/a", "none", "nothing", "no", "yes", "maybe", "unsure", "refusal", "don't know", "not applicable", "non-answer"]):
            non_answer_themes.append(theme_id)
    
    # Common response patterns mapped to actual theme categories
    common_patterns = {
        # Positive responses
        "good": positive_themes, "great": positive_themes, "excellent": positive_themes, "amazing": positive_themes,
        "love": positive_themes, "perfect": positive_themes, "wonderful": positive_themes, "fantastic": positive_themes,
        "awesome": positive_themes, "brilliant": positive_themes, "outstanding": positive_themes,
        
        # Negative responses  
        "bad": negative_themes, "terrible": negative_themes, "awful": negative_themes, "hate": negative_themes,
        "worst": negative_themes, "horrible": negative_themes, "disappointed": negative_themes,
        "disgusting": negative_themes, "pathetic": negative_themes, "useless": negative_themes,
        
        # Neutral responses
        "okay": neutral_themes, "ok": neutral_themes, "fine": neutral_themes, "average": neutral_themes,
        "decent": neutral_themes, "alright": neutral_themes, "mediocre": neutral_themes,
        
        # Non-answers
        "n/a": non_answer_themes, "none": non_answer_themes, "nothing": non_answer_themes, "no": non_answer_themes,
        "yes": non_answer_themes, "maybe": non_answer_themes, "unsure": non_answer_themes,
        "don't know": non_answer_themes, "not sure": non_answer_themes, "no idea": non_answer_themes
    }
    
    fast_assignments = []
    processed_count = 0
    
    for row in unique_rows:
        text = row["text"].lower().strip()
        
        # Check for exact matches first
        if text in common_patterns:
            matching_themes = common_patterns[text]
            if matching_themes:  # If we found matching themes
                # Use the first matching theme with high confidence
                theme_id = matching_themes[0]
                fast_assignments.append({
                    "idx": row["idx"],
                    "assignments": [{"theme_id": theme_id, "confidence": 0.9}],
                    "rationale": "Fast pattern match"
                })
                processed_count += 1
                continue
        
        # Check for partial matches in short responses
        matched = False
        if len(text) < 30:  # Only for short responses
            for pattern, matching_themes in common_patterns.items():
                if pattern in text and matching_themes:
                    theme_id = matching_themes[0]
                    fast_assignments.append({
                        "idx": row["idx"],
                        "assignments": [{"theme_id": theme_id, "confidence": 0.8}],  # Lower confidence for partial match
                        "rationale": "Fast partial pattern match"
                    })
                    processed_count += 1
                    matched = True
                    break
        
        if not matched:
            # Keep original row for AI processing
            fast_assignments.append(row)
    
    # Log the optimization results
    total_rows = len(unique_rows)
    if processed_count > 0:
        st.info(f"⚡ Fast-assigned {processed_count}/{total_rows} responses ({processed_count/total_rows*100:.1f}%)")
    
    return fast_assignments


def slim_theme_for_assignment(theme_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Create a clean theme dictionary for assignment - remove approx_pct to avoid confusion"""
    majors = []
    for m in theme_dict.get("major_themes", []):
        clean_major = {
            "id": m["id"],
            "label": m["label"],
            "definition": m.get("definition", ""),
            "subs": []
        }
        for s in m.get("subs", []):
            clean_sub = {
                "id": s["id"],
                "label": s["label"],
                "definition": s.get("definition", ""),
                "examples": s.get("examples", [])
            }
            clean_major["subs"].append(clean_sub)
        majors.append(clean_major)
    
    # Sort themes to put Non-answer last to avoid defaulting to it
    majors_sorted = []
    non_answer_themes = []
    
    for major in majors:
        if "non-answer" in major.get("label", "").lower():
            non_answer_themes.append(major)
        else:
            majors_sorted.append(major)
    
    # Put non-answer themes at the end
    majors_sorted.extend(non_answer_themes)
    
    return {"major_themes": majors_sorted}


def assign_codes_two_stage(
    client: OpenAI,
    model: str,
    theme_dict: Dict[str, Any],
    rows: List[Dict[str, Any]],
    max_codes: int,
    seed: int | None,
    low_thresh: float,
    top_k: int,
    embedding_model: str,
    progress_bar=None,
    status_text=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    limiter = get_rate_limiter()
    cache = get_llm_cache()
    cache_stats = get_cache_stats()
    guidance_text = st.session_state.get(f"question_guidance::{text_col}", "").strip() if "text_col" in globals() else ""
    return pipeline_assign_codes_two_stage(
        client,
        model,
        theme_dict,
        rows,
        max_codes,
        seed or 42,
        low_thresh,
        top_k,
        embedding_model,
        limiter=limiter,
        cache=cache,
        cache_stats=cache_stats,
        on_status=status_text.text if status_text else None,
        on_progress=progress_bar.progress if progress_bar else None,
        guidance_text=guidance_text,
    )


def expand_deduplicated_results(unique_assignments: List[Dict[str, Any]], response_to_indices: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    """Expand deduplicated results back to all original responses"""
    # The unique_assignments contain assignments with idx values that correspond to 
    # the position of unique responses (0, 1, 2, ...), not the original row indices
    
    # Create a list of unique texts in the same order as returned by deduplicate_responses
    unique_texts = list(response_to_indices.keys())
    
    # Create mapping from unique response position to assignment
    idx_to_assignment = {assignment["idx"]: assignment for assignment in unique_assignments}
    
    # Create mapping from text to assignment
    text_to_assignment = {}
    for unique_idx, assignment in idx_to_assignment.items():
        if unique_idx < len(unique_texts):
            text = unique_texts[unique_idx]
            text_to_assignment[text] = assignment
    
    # Expand to all original responses
    expanded_assignments = []
    for text, original_indices in response_to_indices.items():
        if text in text_to_assignment:
            base_assignment = text_to_assignment[text]
            # Create assignments for all original indices with this text
            for original_idx in original_indices:
                # Deep copy to avoid shared mutable lists across duplicates
                import copy
                expanded_assignment = copy.deepcopy(base_assignment)
                expanded_assignment["idx"] = original_idx
                expanded_assignments.append(expanded_assignment)
    
    # Sort by original index to maintain order
    expanded_assignments.sort(key=lambda x: x["idx"])
    return expanded_assignments

def verify_low_confidence(
            client: OpenAI,
            model: str,
            theme_dict: Dict[str, Any],
            flagged: List[Dict[str, Any]],
            low_thresh: float,
            max_codes: int,
            seed: int | None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Verify low confidence assignments"""
    if not flagged:
        return [], {"prompt_tokens": 0, "completion_tokens": 0}

    chunk_size = 5
    total_chunks = math.ceil(len(flagged) / chunk_size)

    # Constrain verification outputs to valid theme IDs
    _allowed_ids = allowed_subtheme_ids(theme_dict)
    schema = make_assignments_schema(_allowed_ids, max_codes=max_codes)

    # Prepare all chunks for parallel processing
    all_chunks = []
    for i in range(0, len(flagged), chunk_size):
        chunk_flagged = flagged[i:i + chunk_size]
        all_chunks.append(chunk_flagged)
    
    def process_verification_chunk(chunk_flagged):
        """Process a single chunk of flagged responses"""
        theme_json = json.dumps(slim_theme_for_assignment(theme_dict))
        flagged_json = json.dumps(chunk_flagged)
        guidance_text = st.session_state.get(f"question_guidance::{text_col}", "").strip() if "text_col" in globals() else ""
        guidance_block = f"\n\nGuidance (must follow):\n{guidance_text}\n" if guidance_text else ""
        user = VERIFY_USER_TEMPLATE.format(low_thresh=low_thresh, theme_json=theme_json, flagged_json=flagged_json) + guidance_block
        
        def make_request():
            return oai_json_completion(client, model, VERIFY_SYSTEM, user, seed, schema)

        return retry_with_backoff(make_request)
    
    # Use parallel processing for verification
    max_workers = min(5, len(all_chunks))
    
    all_verified = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {executor.submit(process_verification_chunk, chunk): i for i, chunk in enumerate(all_chunks)}
        
        completed = 0
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                data, usage = future.result()
                
                # Parse response
                if isinstance(data, dict) and "results" in data:
                    chunk_verified = data["results"]
                else:
                    chunk_verified = data if isinstance(data, list) else []
                
                all_verified.extend(chunk_verified)
                
                # Accumulate usage
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                
                completed += 1
                
            except Exception as e:
                st.error(f"Error verifying chunk: {str(e)}")
                # Continue with other chunks
    
    return all_verified, total_usage


# ------------------------------
# Theme helpers
# ------------------------------

def analyze_theme_distribution(coded_df: pd.DataFrame, tiny_threshold: float, theme_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze theme distribution and identify potential issues with outlier handling"""
    total_responses = len(coded_df)
    
    # Count responses in different categories
    other_themes = coded_df[coded_df[f"{question_label}_MinorTheme1"].str.contains("Other", case=False, na=False)]
    not_applicable = coded_df[coded_df[f"{question_label}_MinorTheme1"].str.contains("Not applicable", case=False, na=False)]
    manual_review_needed = coded_df.get("ManualReview", pd.Series([False] * len(coded_df), dtype=bool))
    
    other_count = len(other_themes)
    not_applicable_count = len(not_applicable)
    manual_review_count = manual_review_needed.sum() if hasattr(manual_review_needed, 'sum') else 0
    
    other_percent = (other_count / total_responses * 100) if total_responses > 0 else 0
    not_applicable_percent = (not_applicable_count / total_responses * 100) if total_responses > 0 else 0
    manual_review_percent = (manual_review_count / total_responses * 100) if total_responses > 0 else 0
    
    # Identify themes with very low counts (potential candidates for "Other")
    theme_counts = coded_df[f"{question_label}_MinorTheme1"].value_counts()
    tiny_themes = theme_counts[theme_counts == 1]  # Single-response themes
    tiny_theme_count = len(tiny_themes)
    tiny_theme_percent = (tiny_theme_count / total_responses * 100) if total_responses > 0 else 0
    
    # Check coverage estimates vs actual if theme_dict provided
    coverage_analysis = {}
    if theme_dict:
        coverage_analysis = analyze_coverage_accuracy(coded_df, theme_dict)
    
    # Analysis results
    analysis = {
        "total_responses": total_responses,
        "other_count": other_count,
        "other_percent": other_percent,
        "not_applicable_count": not_applicable_count,
        "not_applicable_percent": not_applicable_percent,
        "manual_review_count": manual_review_count,
        "manual_review_percent": manual_review_percent,
        "tiny_theme_count": tiny_theme_count,
        "tiny_theme_percent": tiny_theme_percent,
        "tiny_theme_names": tiny_themes.index.tolist(),
        "threshold_exceeded": tiny_theme_percent > tiny_threshold,
        "coverage_analysis": coverage_analysis,
        "recommendations": []
    }
    
    # Generate recommendations
    if not_applicable_percent > 5.0:
        analysis["recommendations"].append(
            f"High 'Not applicable' rate ({not_applicable_percent:.1f}%) - consider if these responses are truly non-applicable or should be in specific themes"
        )
    
    if other_percent > 10.0:
        analysis["recommendations"].append(
            f"High 'Other' category usage ({other_percent:.1f}%) - these responses likely reveal specific themes that should be explicitly defined rather than grouped as 'Other'"
        )
    
    if manual_review_percent > 2.0:
        analysis["recommendations"].append(
            f"High manual review rate ({manual_review_percent:.1f}%) - consider refining themes or assignment logic"
        )
    
    if tiny_theme_percent > tiny_threshold and other_percent < 5.0:
        analysis["recommendations"].append(
            f"Many single-response themes ({tiny_theme_count} themes, {tiny_theme_percent:.1f}%) but low 'Other' usage - good theme specificity"
        )
    elif tiny_theme_percent > tiny_threshold:
        analysis["recommendations"].append(
            f"Many single-response themes ({tiny_theme_count} themes, {tiny_theme_percent:.1f}%) - consider if some can be consolidated"
        )
    
    return analysis


def auto_merge_tiny_subthemes(
    theme_dict: Dict[str, Any],
    assigned_raw: List[Dict[str, Any]],
    tiny_threshold_pct: float,
    other_usage_threshold_pct: float = 15.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Merge tiny sub-themes into an 'Other' bucket per major theme."""
    if not theme_dict or not assigned_raw:
        return theme_dict, assigned_raw, {"merged_count": 0, "other_exceeded": False, "other_share_by_major": {}}

    def norm_alpha(x: str) -> str:
        return re.sub(r"[^a-z]+", "", (x or "").lower())

    majors = theme_dict.get("major_themes", []) or []
    sub_to_major: Dict[str, str] = {}
    sub_label: Dict[str, str] = {}
    for m in majors:
        for s in m.get("subs", []) or []:
            sid = s.get("id")
            if sid:
                sub_to_major[sid] = m.get("id")
                sub_label[sid] = s.get("label", "")

    # Count primary assignments
    sub_counts: Dict[str, int] = {}
    major_counts: Dict[str, int] = {}
    for item in assigned_raw:
        assigns = item.get("assignments", [])
        if not assigns:
            continue
        assigns = sorted(assigns, key=lambda a: a.get("confidence", 0.0), reverse=True)
        top = assigns[0]
        tid = top.get("theme_id")
        if not tid:
            continue
        mid = sub_to_major.get(tid) or tid
        sub_counts[tid] = sub_counts.get(tid, 0) + 1
        major_counts[mid] = major_counts.get(mid, 0) + 1

    tiny_ids: set[str] = set()
    tiny_to_other: Dict[str, str] = {}
    merged_count = 0

    for m in majors:
        if norm_alpha(m.get("label", "")) == "nonanswer":
            continue
        mid = m.get("id")
        total = major_counts.get(mid, 0)
        if total <= 0:
            continue

        subs = m.get("subs", []) or []
        other_sub = next((s for s in subs if "other" in (s.get("label", "") or "").lower()), None)
        if not other_sub:
            # Create an "Other" sub-theme
            used_nums = set()
            for s in subs:
                sid = str(s.get("id", ""))
                mm = re.match(rf"^{re.escape(mid)}\.(\d+)$", sid)
                if mm:
                    used_nums.add(int(mm.group(1)))
            next_n = 1
            while next_n in used_nums:
                next_n += 1
            other_sub = {
                "id": f"{mid}.{next_n}",
                "label": "Other",
                "definition": "Miscellaneous responses that do not fit other sub-themes.",
                "approx_pct": 0.0,
                "examples": [],
            }
            subs.append(other_sub)

        other_id = other_sub.get("id")

        # Determine tiny sub-themes within this major
        for s in list(subs):
            sid = s.get("id")
            if not sid or sid == other_id:
                continue
            if "other" in (s.get("label", "") or "").lower():
                continue
            pct = (sub_counts.get(sid, 0) / total) * 100
            if pct < tiny_threshold_pct:
                tiny_ids.add(sid)
                tiny_to_other[sid] = other_id

        if tiny_ids:
            # Remove tiny sub-themes from the dictionary
            m["subs"] = [s for s in subs if s.get("id") not in tiny_ids or s.get("id") == other_id]

    # Update assignments: swap tiny IDs to Other
    updated_assignments: List[Dict[str, Any]] = []
    for item in assigned_raw:
        assigns = item.get("assignments", [])
        if not assigns:
            updated_assignments.append(item)
            continue
        new_assigns = []
        for a in assigns:
            tid = a.get("theme_id")
            if tid in tiny_to_other:
                a = dict(a)
                a["theme_id"] = tiny_to_other[tid]
            new_assigns.append(a)
        # Deduplicate by theme_id keeping max confidence
        by_id: Dict[str, float] = {}
        for a in new_assigns:
            tid = a.get("theme_id")
            if not tid:
                continue
            conf = float(a.get("confidence", 0.0))
            by_id[tid] = max(conf, by_id.get(tid, 0.0))
        deduped = [{"theme_id": k, "confidence": v} for k, v in by_id.items()]
        deduped.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        updated_item = dict(item)
        updated_item["assignments"] = deduped
        updated_assignments.append(updated_item)

    merged_count = len(tiny_to_other)

    # Compute "Other" share by major after merge
    other_share_by_major: Dict[str, float] = {}
    other_exceeded = False
    for m in majors:
        if norm_alpha(m.get("label", "")) == "nonanswer":
            continue
        mid = m.get("id")
        total = major_counts.get(mid, 0)
        if total <= 0:
            continue
        other_sub = next((s for s in (m.get("subs") or []) if "other" in (s.get("label", "") or "").lower()), None)
        if not other_sub:
            continue
        other_id = other_sub.get("id")
        other_count = sub_counts.get(other_id, 0)
        share = (other_count / total) * 100
        other_share_by_major[mid] = share
        if share > other_usage_threshold_pct:
            other_exceeded = True

    report = {
        "merged_count": merged_count,
        "other_exceeded": other_exceeded,
        "other_share_by_major": other_share_by_major,
        "tiny_threshold_pct": tiny_threshold_pct,
        "other_usage_threshold_pct": other_usage_threshold_pct,
    }
    return theme_dict, updated_assignments, report


def analyze_coverage_accuracy(coded_df: pd.DataFrame, theme_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compare AI-estimated coverage percentages with actual results"""
    if not theme_dict or "major_themes" not in theme_dict:
        return {}
    
    total_responses = len(coded_df)
    coverage_results = []
    
    # Build theme label to estimated percentage mapping
    theme_estimates = {}
    for major in theme_dict["major_themes"]:
        if major.get("approx_pct"):
            theme_estimates[major.get("label", "")] = major.get("approx_pct", 0.0)
        for sub in major.get("subs", []):
            if sub.get("approx_pct"):
                theme_estimates[sub.get("label", "")] = sub.get("approx_pct", 0.0)
    
    # Compare with actual counts
    actual_counts = coded_df[f"{question_label}_MinorTheme1"].value_counts()
    
    for theme_label, estimated_pct in theme_estimates.items():
        actual_count = actual_counts.get(theme_label, 0)
        actual_pct = actual_count / total_responses if total_responses > 0 else 0
        
        coverage_results.append({
            "theme": theme_label,
            "estimated_pct": estimated_pct,
            "actual_pct": actual_pct,
            "accuracy": abs(estimated_pct - actual_pct),
            "actual_count": actual_count
        })
    
    # Sort by accuracy (best estimates first)
    coverage_results.sort(key=lambda x: x["accuracy"])
    
    return {
        "theme_accuracy": coverage_results,
        "avg_accuracy": sum(r["accuracy"] for r in coverage_results) / len(coverage_results) if coverage_results else 0,
        "best_estimates": coverage_results[:3] if len(coverage_results) >= 3 else coverage_results,
        "worst_estimates": coverage_results[-3:] if len(coverage_results) >= 3 else []
    }


def flatten_theme_dict(theme_dict: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for m in theme_dict.get("major_themes", []):
        rows.append({
            "ThemeID": m.get("id"),
            "Level": "Major",
            "ParentThemeID": "",
            "Label": m.get("label"),
            "ShortDefinition": m.get("definition", ""),
            "ApproxPct": m.get("approx_pct", 0.0),
            "ExampleQuotes": "",
        })
        for s in m.get("subs", []):
            rows.append({
                "ThemeID": s.get("id"),
                "Level": "Sub",
                "ParentThemeID": m.get("id"),
                "Label": s.get("label"),
                "ShortDefinition": s.get("definition", ""),
                "ApproxPct": s.get("approx_pct", 0.0),
                "ExampleQuotes": "; ".join(s.get("examples", [])[:3]),
            })
    return pd.DataFrame(rows)


def map_theme_id_to_major(theme_df: pd.DataFrame) -> Dict[str, str]:
    major_of = {}
    for _, r in theme_df.iterrows():
        if r["Level"] == "Major":
            # Major maps to itself
            major_of[r["ThemeID"]] = r["ThemeID"]
        else:
            major_of[r["ThemeID"]] = r["ParentThemeID"]
    return major_of


def governance_log_to_df(change_log: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in change_log or []:
        rows.append({
            "Action": entry.get("action", ""),
            "FromIDs": "; ".join(entry.get("from_ids", []) or []),
            "ToIDs": "; ".join(entry.get("to_ids", []) or []),
            "Reason": entry.get("reason", ""),
        })
    return pd.DataFrame(rows)


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(
    page_title="Open-End Coding",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #427043;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #1976d2;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊Open-End Coding</h1>
    <p>Upload open‑ended survey responses, generate themes, and export coded data.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model = "gpt-5"
    seed = 42  # Hard-coded for deterministic results
    st.info("Current Mode: GPT-5 for All Steps")
    redact_pii_enabled = True
    allow_multicode = st.toggle("Multi‑coding", value=True)

    max_codes = 3
    single_or_multi = "Multi" if allow_multicode else "Single"

    embedding_model = st.selectbox(
        "Embedding model",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )
    top_k = st.slider("Candidate shortlist size (top_k)", 3, 12, 6, 1)

    low_thresh = st.slider("Low confidence threshold", 0.0, 1.0, 0.60, 0.01)
    # Tiny theme threshold is auto-calculated after assignments for consistency
    tiny_threshold = None


uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

if uploaded is None:
    st.markdown("""
    <div class="info-box">
        <h4>📁 Ready to get started?</h4>
        <p>Upload a CSV or XLSX file with your open-ended survey responses. The file should have:</p>
        <ul>
            <li>One column per question</li>
            <li>One row per response</li>
            <li>Optional ID columns for tracking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load file
if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    df = pd.read_excel(uploaded)

# Clean column headers
df.columns = [str(c).strip() for c in df.columns]

# Ask for question column
text_col = st.selectbox("Select the open‑end column", options=df.columns.tolist())

# Reset downstream results if the uploaded file or selected column changed
_input_sig = (uploaded.name, getattr(uploaded, "size", None), text_col)
if st.session_state.get("_input_sig") != _input_sig:
    for k in ("assigned_raw", "theme_validation", "_usage_totals", "_auto_verified", "theme_governance_log", "_cache_stats"):
        st.session_state.pop(k, None)
    st.session_state["_input_sig"] = _input_sig

# Optional ID passthrough
id_cols_guess = [c for c in df.columns if c.lower() in {"id", "respondent_id", "record id", "record_id", "transaction id", "transaction_id", "uuid"}]
pass_id_cols = st.multiselect("ID columns to carry through", options=df.columns.tolist(), default=id_cols_guess)


# Prepare series
ser = df[text_col].map(clean_text)
ser_ai = ser.map(redact_pii) if redact_pii_enabled else ser

# Build unique set with frequency weights (AI-safe text) but preserve order for output mapping
value_counts = ser_ai.value_counts(dropna=False)
unique_texts = value_counts.index.tolist()
unique_freqs = value_counts.values.tolist()

# Initialize OpenAI client
client = get_openai_client()

# Use the selected column header as question context
question_text = text_col

# Infer a compact question label (qAge, qGender, etc.)
label_key = f"question_label::{text_col}"
if st.session_state.get(label_key) is None:
    with st.spinner("Inferring question label..."):
        st.session_state[label_key] = infer_question_label(client, model, question_text, seed)
question_label = st.session_state[label_key]

# Run signature + prompt versions
dataset_sig = dataset_signature(ser_ai.fillna("").astype(str).tolist(), question_text)
prompt_versions = {
    "theme_discovery": compute_prompt_version(PIPE_THEME_DISCOVERY_SYSTEM, PIPE_THEME_DISCOVERY_USER),
    "governance": compute_prompt_version(PIPE_GOVERNANCE_SYSTEM, PIPE_GOVERNANCE_USER_TEMPLATE),
    "assignment_stage2": compute_prompt_version(PIPE_ASSIGNMENT_STAGE2_SYSTEM, PIPE_ASSIGNMENT_STAGE2_USER_TEMPLATE),
}
run_settings = {
    "model": model,
    "seed": seed,
    "max_codes": max_codes if allow_multicode else 1,
    "allow_multicode": allow_multicode,
    "low_thresh": float(low_thresh),
    "top_k": int(top_k),
    "embedding_model": embedding_model,
}
run_id = compute_run_id(dataset_sig, model, seed, prompt_versions, run_settings)
st.session_state["run_id"] = run_id

st.divider()
st.subheader("Theme discovery")

# Theme management options
col1, col2 = st.columns([2, 1])
with col1:
    st.write("**Choose theme source:**")
with col2:
    theme_source = st.radio(
        "Theme source",
        ["Generate new themes", "Upload existing themes", "Add coding context"],
        horizontal=True,
        label_visibility="collapsed",
    )

# Theme upload functionality
if theme_source == "Upload existing themes":
    uploaded_theme_file = st.file_uploader("Upload theme dictionary (JSON or XLSX)", type=["json", "xlsx"],
                                           key="theme_upload")

    if uploaded_theme_file is not None:
        try:
            if uploaded_theme_file.name.lower().endswith('.json'):
                theme_dict = json.load(uploaded_theme_file)
            else:  # XLSX
                theme_df_upload = pd.read_excel(uploaded_theme_file)
                # Convert XLSX to theme dictionary format
                theme_dict = {"major_themes": []}
                current_major = None

                for _, row in theme_df_upload.iterrows():
                    if row.get("Level") == "Major":
                        current_major = {
                            "id": row.get("ThemeID", ""),
                            "label": row.get("Label", ""),
                            "definition": row.get("ShortDefinition", ""),
                            "subs": []
                        }
                        theme_dict["major_themes"].append(current_major)
                    elif row.get("Level") == "Sub" and current_major:
                        sub_theme = {
                            "id": row.get("ThemeID", ""),
                            "label": row.get("Label", ""),
                            "definition": row.get("ShortDefinition", ""),
                            "examples": row.get("ExampleQuotes", "").split("; ") if pd.notna(
                                row.get("ExampleQuotes")) else []
                        }
                        current_major["subs"].append(sub_theme)

            theme_dict = ensure_nonanswer_theme(theme_dict)
            st.session_state["theme_dict"] = theme_dict
            # Clear any prior assignments tied to a different codebook
            st.session_state.pop("assigned_raw", None)
            st.session_state.pop("theme_validation", None)
            st.success("Theme dictionary uploaded successfully!")

        except Exception as e:
            st.error(f"Error loading theme file: {str(e)}")

        if "theme_dict" in st.session_state:
            st.write("**Current theme dictionary:**")
            theme_df = flatten_theme_dict(st.session_state["theme_dict"])
            st.dataframe(theme_df, width="stretch")

            if st.button("Apply uploaded themes", type="primary"):
                # Assign themes with progress
                apply_progress = st.progress(0)
                apply_status = st.empty()

                question_context = detect_question_type(question_text) if question_text else {"type": "general",
                                                                                              "focus": "General analysis",
                                                                                              "priority_themes": []}
                st.session_state["theme_validation"] = validate_theme_quality(st.session_state["theme_dict"],
                                                                              question_context)

                rows_payload = [
                    {"idx": int(i), "text": t}
                    for i, t in enumerate(ser_ai.fillna("").astype(str).tolist())
                ]

                assigned, usage_assign = assign_codes_with_progress(
                    client, model, st.session_state["theme_dict"], rows_payload,
                    max_codes if allow_multicode else 1, seed, low_thresh, top_k, embedding_model, apply_progress, apply_status
                )
                st.session_state["assigned_raw"] = assigned
                st.success("✅ Themes applied to responses. Scroll down to review and export.")

        if st.button("Clear uploaded themes"):
            for k in ("theme_dict", "assigned_raw", "theme_validation"):
                st.session_state.pop(k, None)
            st.rerun()

# Guidance input section
guidance_key = f"question_guidance::{text_col}"
if theme_source == "Add coding context":
    st.write("**Provide guidance for theme generation and/or assignment:**")
    guidance_text = st.text_area(
        "Coding guidance",
        value=st.session_state.get(guidance_key, ""),
        help="Add question-level rules or constraints (e.g., classify only US destinations).",
        height=120,
    )
    if guidance_text.strip():
        st.session_state[guidance_key] = guidance_text.strip()
        st.success("Guidance saved for this question.")
    else:
        st.warning("Guidance is empty; no additional rules will be applied.")

    st.write("**Target type (optional):**")
    target_options = {
        "Auto (from guidance)": "auto",
        "City/area": "city",
        "Country": "country",
        "US state": "state",
        "Brand": "brand",
        "Destination": "destination",
        "Adjective/descriptor": "adjective",
        "One-word response": "one_word",
    }
    selected_label = st.selectbox(
        "Target type",
        list(target_options.keys()),
        index=0,
    )
    st.session_state["guidance_target_type"] = target_options[selected_label]

    st.write("**Location normalization (Nominatim):**")
    normalize_locations = st.checkbox(
        "Normalize city/area labels using external lookup",
        value=bool(st.session_state.get("normalize_locations", True)),
        help="Uses OpenStreetMap Nominatim to standardize labels (e.g., City, ST).",
        key="normalize_locations",
    )
    if normalize_locations:
        default_ua = ""
        try:
            default_ua = st.secrets.get("NOMINATIM_USER_AGENT", "")
        except Exception:
            default_ua = ""
        if not default_ua:
            default_ua = os.getenv("NOMINATIM_USER_AGENT", "OpenEndCoding/1.0 (contact: local)")
        st.session_state["nominatim_user_agent"] = st.text_input(
            "Nominatim User-Agent",
            value=st.session_state.get("nominatim_user_agent", default_ua),
            help="Required by Nominatim. Provide a descriptive User-Agent.",
        )
        if not (st.session_state.get("nominatim_user_agent", "").strip()):
            st.warning("Nominatim User-Agent is required; external normalization will be skipped.")

active_guidance = st.session_state.get(guidance_key, "").strip()
if active_guidance:
    st.caption(f"**Active guidance:** {active_guidance}")
    # #region agent log
    try:
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H1",
                "location": "app_v1.py:guidance:active",
                "message": "Active guidance set",
                "data": {"guidance_len": len(active_guidance)},
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion

if theme_source == "Add coding context":
    run_settings["normalize_locations"] = bool(st.session_state.get("normalize_locations", False))
    run_settings["geocode_provider"] = "nominatim" if run_settings["normalize_locations"] else "none"
    run_settings["guidance_target_type"] = st.session_state.get("guidance_target_type", "auto")
    run_id = compute_run_id(dataset_sig, model, seed, prompt_versions, run_settings)
    st.session_state["run_id"] = run_id

# Theme generation section
show_generate = theme_source == "Generate new themes" or (
    theme_source == "Add coding context" and bool(active_guidance)
)
if show_generate:
    # #region agent log
    try:
        with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "H2",
                "location": "app_v1.py:theme_gen:entry",
                "message": "Theme generation section visible",
                "data": {"show_generate": True, "guidance_len": len(active_guidance or "")},
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion
    # Cost estimation before generation
    pricing_table = {
        "gpt-5": {"prompt_per_1k": 0.005, "completion_per_1k": 0.015},  # Estimated pricing
        "gpt-4o": {"prompt_per_1k": 0.005, "completion_per_1k": 0.015},
        "gpt-4o-mini": {"prompt_per_1k": 0.0005, "completion_per_1k": 0.0015},
        "gpt-4-turbo": {"prompt_per_1k": 0.01, "completion_per_1k": 0.03},
    }

    # Estimate costs for theme generation (GPT-5)
    pricing = pricing_table.get(model, {"prompt_per_1k": 0.005, "completion_per_1k": 0.015})
    
    # More accurate token estimation (accounting for pre-filtering)
    # Pre-filter the same way as in build_theme_frame
    filtered_for_estimation = [
        {"text": t, "weight": int(w)} for t, w in zip(unique_texts, unique_freqs)
        if t and not is_empty_like(t)
    ]
    estimated_prompt_tokens = estimate_tokens(json.dumps(filtered_for_estimation))
    estimated_completion_tokens = 2000  # Rough estimate for theme generation
    
    # Keep estimation internal; no extra UI noise

    if st.button("Process Themes", type="primary"):
        # #region agent log
        try:
            with open(r"c:\Users\apier\PycharmProjects\OpenEndCoding\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H3",
                    "location": "app_v1.py:theme_gen:click",
                    "message": "Process Themes clicked",
                    "data": {
                        "guidance_len": len(st.session_state.get(guidance_key, "") or ""),
                        "theme_source": theme_source,
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        # Start the overall timer
        overall_start_time = time.time()
        
        # Create progress containers
        progress_container = st.container()
        status_container = st.container()
        
        # Create timer display
        timer_container = st.container()
        with timer_container:
            st.markdown("### ⏱️ Processing Timer")
            timer_col1, timer_col2, timer_col3, timer_col4 = st.columns(4)
            with timer_col1:
                total_timer = st.empty()
            with timer_col2:
                theme_timer = st.empty()
            with timer_col3:
                assign_timer = st.empty()
            with timer_col4:
                build_timer = st.empty()
        
        with progress_container:
            st.subheader("Processing Progress")
            
            # Step 1: Generate themes with progress
            st.write("🎯 **Generating Hierarchical Themes**")
            theme_progress = st.progress(0)
            theme_status = st.empty()
            
            theme_start_time = time.time()
            theme_status.text("Initializing theme generation...")
            theme_progress.progress(10)
            
            # Update timer display
            total_timer.metric("⏱️ Total Time", "0:00")
            theme_timer.metric("🎯 Theme Gen", "Running...")
            assign_timer.metric("🏷️ Assignment", "Waiting...")
            build_timer.metric("📊 Building", "Waiting...")
            
            # Detect question type for optimized theme generation
            question_context = detect_question_type(question_text) if question_text else {"type": "general", "focus": "General analysis", "priority_themes": []}
            
            # Display question type insights only in logs
            
            theme_dict, usage_theme = build_theme_frame_with_progress(
                client,
                model,
                unique_texts,
                unique_freqs,
                seed,
                theme_progress,
                theme_status,
                question_context,
            )
            theme_dict = ensure_nonanswer_theme(theme_dict)

            try:
                theme_dict, change_log, usage_gov = govern_theme_dict(
                    client,
                    model,
                    theme_dict,
                    question_context,
                    seed,
                )
                st.session_state["theme_governance_log"] = change_log
                usage_theme["prompt_tokens"] += usage_gov.get("prompt_tokens", 0)
                usage_theme["completion_tokens"] += usage_gov.get("completion_tokens", 0)
                if change_log:
                    pass
                else:
                    pass
            except Exception as e:
                st.error(f"Theme governance failed: {str(e)}")
                st.stop()

            st.session_state["theme_dict"] = theme_dict
            
            # Validate theme quality
            theme_validation = validate_theme_quality(theme_dict, question_context)
            st.session_state["theme_validation"] = theme_validation
            
            theme_end_time = time.time()
            theme_duration = theme_end_time - theme_start_time
            
            theme_progress.progress(100)
            theme_status.text("✅ Theme generation complete!")
            
            # Update timers
            total_elapsed = time.time() - overall_start_time
            total_timer.metric("⏱️ Total Time", f"{int(total_elapsed//60)}:{int(total_elapsed%60):02d}")
            theme_timer.metric("🎯 Theme Gen", f"{int(theme_duration//60)}:{int(theme_duration%60):02d}")
            
            # Step 2: Assign themes with progress
            st.write("🏷️ **Assigning Themes to Responses**")
            assign_progress = st.progress(0)
            assign_status = st.empty()
            
            assign_start_time = time.time()
            assign_status.text("Preparing assignment data...")
            assign_progress.progress(5)
            
            # Update timer display
            assign_timer.metric("🏷️ Assignment", "Running...")
            
            rows_payload = [
                {"idx": int(i), "text": t}
                for i, t in enumerate(ser_ai.fillna("").astype(str).tolist())
            ]

            assigned, usage_assign = assign_codes_with_progress(
                client,
                model,
                theme_dict,
                rows_payload,
                max_codes if allow_multicode else 1,
                seed,
                low_thresh,
                top_k,
                embedding_model,
                assign_progress,
                assign_status,
            )
            st.session_state["assigned_raw"] = assigned
            
            assign_end_time = time.time()
            assign_duration = assign_end_time - assign_start_time
            
            assign_progress.progress(100)
            assign_status.text("✅ Theme assignment complete!")
            
            # Update timers
            total_elapsed = time.time() - overall_start_time
            total_timer.metric("⏱️ Total Time", f"{int(total_elapsed//60)}:{int(total_elapsed%60):02d}")
            assign_timer.metric("🏷️ Assignment", f"{int(assign_duration//60)}:{int(assign_duration%60):02d}")
            
            # Step 3: Build coded dataframe
            st.write("📊 **Building Coded Dataset**")
            build_progress = st.progress(0)
            build_status = st.empty()
            
            build_start_time = time.time()
            build_status.text("Processing coded data...")
            build_progress.progress(50)
            
            # Update timer display
            build_timer.metric("📊 Building", "Running...")
            
            # This will be done in the main flow below
            build_progress.progress(100)
            build_status.text("✅ Dataset ready for analysis!")
            
            build_end_time = time.time()
            build_duration = build_end_time - build_start_time
            
            # Final timer update
            total_elapsed = time.time() - overall_start_time
            total_timer.metric("⏱️ Total Time", f"{int(total_elapsed//60)}:{int(total_elapsed%60):02d}")
            build_timer.metric("📊 Building", f"{int(build_duration//60)}:{int(build_duration%60):02d}")
        
        # Final summary with timing
        total_minutes = int(total_elapsed // 60)
        total_seconds = int(total_elapsed % 60)
        theme_minutes = int(theme_duration // 60)
        theme_seconds = int(theme_duration % 60)
        assign_minutes = int(assign_duration // 60)
        assign_seconds = int(assign_duration % 60)
        
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Theme Processing Complete!</h4>
            <p><strong>⏱️ Total Processing Time: {total_minutes}:{total_seconds:02d}</strong></p>
            <ul>
                <li>🎯 Theme Generation: {theme_minutes}:{theme_seconds:02d}</li>
                <li>🏷️ Theme Assignment: {assign_minutes}:{assign_seconds:02d}</li>
                <li>📊 Dataset Building: <1 second</li>
            </ul>
            <p>Your themes have been generated and assigned to all responses. Review the results below and use the verification tools if needed.</p>
        </div>
        """, unsafe_allow_html=True)

if "theme_dict" not in st.session_state or "assigned_raw" not in st.session_state:
    st.stop()

# Auto-merge tiny sub-themes into "Other" based on dynamic thresholds
dynamic_thresholds = calculate_dynamic_thresholds(
    st.session_state.get("theme_dict"),
    st.session_state.get("assigned_raw"),
)
tiny_threshold = dynamic_thresholds.get("tiny_threshold", 1.0)
other_usage_threshold = dynamic_thresholds.get("other_usage_threshold", 15.0)

merged_theme_dict, merged_assignments, merge_report = auto_merge_tiny_subthemes(
    st.session_state["theme_dict"],
    st.session_state["assigned_raw"],
    tiny_threshold_pct=tiny_threshold,
    other_usage_threshold_pct=other_usage_threshold,
)
st.session_state["theme_dict"] = merged_theme_dict
st.session_state["assigned_raw"] = merged_assignments
st.session_state["merge_report"] = merge_report

if merge_report.get("merged_count", 0) > 0:
    st.info(
        f"Auto-merged {merge_report['merged_count']} tiny sub-themes into 'Other' "
        f"(threshold: {tiny_threshold:.2f}%)."
    )
if merge_report.get("other_exceeded"):
    st.warning(
        "⚠️ 'Other' share is high in at least one major theme. "
        "Consider regenerating themes with stricter consolidation guidance."
    )

# Show theme quality validation
if "theme_validation" in st.session_state:
    validation = st.session_state["theme_validation"]

# Show theme dictionary
st.write("**Generated Theme Dictionary:**")
theme_df = flatten_theme_dict(st.session_state["theme_dict"])
st.dataframe(theme_df, width="stretch")

# Theme export functionality
st.write("**Export Theme Dictionary:**")
col1, col2 = st.columns(2)

with col1:
    # JSON export
    theme_json = json.dumps(st.session_state["theme_dict"], indent=2)
    st.download_button(
        "📄 Download as JSON",
        data=theme_json,
        file_name=f"theme_dictionary_{today_stamp()}.json",
        mime="application/json",
        help="Download theme dictionary as JSON for easy import back into the tool"
    )

with col2:
    # XLSX export
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        theme_df.to_excel(writer, sheet_name="Theme Dictionary", index=False)
    
    st.download_button(
        "📊 Download as XLSX",
        data=buf.getvalue(),
        file_name=f"theme_dictionary_{today_stamp()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download theme dictionary as XLSX for easy import back into the tool"
    )
# (tip removed for cleaner UI)
# Comprehensive export (themes + coded data)
st.write("**Complete Export (Themes + Coded Data):**")
comprehensive_buf = io.BytesIO()
with pd.ExcelWriter(comprehensive_buf, engine="xlsxwriter") as writer:
    # Add theme dictionary sheet
    theme_df.to_excel(writer, sheet_name="Theme Dictionary", index=False)
    
    governance_log = st.session_state.get("theme_governance_log") or []
    if governance_log:
        governance_df = governance_log_to_df(governance_log)
        governance_df.to_excel(writer, sheet_name="Theme Governance Log", index=False)
    
    # Add coded data sheet (we'll need to build this first)
    if "assigned_raw" in st.session_state:
        # Build a preview of coded data for the comprehensive export
        assign_map = {x["idx"]: x for x in st.session_state["assigned_raw"]}
        major_map = map_theme_id_to_major(theme_df)
        label_map = {r["ThemeID"]: r["Label"] for _, r in theme_df.iterrows()}
        theme_levels = {r["ThemeID"]: r["Level"] for _, r in theme_df.iterrows()}
        
        # Create a sample of coded data (first 100 rows for export)
        sample_coded_rows = []
        for i in range(min(100, len(assign_map))):
            item = assign_map.get(i, {"assignments": [], "rationale": ""})
            assigns = item.get("assignments", [])
            if assigns:
                primary_theme_id = assigns[0].get("theme_id", "")
                primary_major = major_map.get(primary_theme_id, "")
                primary_sub = primary_theme_id if theme_levels.get(primary_theme_id) == "Sub" else ""
                
                sample_coded_rows.append({
                    "Response_Index": i,
                    "Question": text_col,
                    "QuestionLabel": question_label,
                    "MajorTheme": label_map.get(primary_major, ""),
                    "SubTheme": label_map.get(primary_theme_id, "") if primary_sub else "",
                })
        
        if sample_coded_rows:
            sample_coded_df = pd.DataFrame(sample_coded_rows)
            sample_coded_df.to_excel(writer, sheet_name="Sample Coded Data", index=False)

    # Add question mapping sheet
    question_map_df = pd.DataFrame([{"QuestionLabel": question_label, "QuestionText": text_col, "RunId": st.session_state.get("run_id", "")}])
    question_map_df.to_excel(writer, sheet_name="Question Map", index=False)

st.download_button(
    "📦 Download Complete Package",
    data=comprehensive_buf.getvalue(),
    file_name=f"complete_thematic_analysis_{today_stamp()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Download complete package with themes and sample coded data"
)

# Build coded DataFrame first
assign_map = {x["idx"]: x for x in st.session_state["assigned_raw"]}

# Theme map helpers
major_map = map_theme_id_to_major(theme_df)
label_map = {r["ThemeID"]: r["Label"] for _, r in theme_df.iterrows()}
parent_map = {r["ThemeID"]: r["ParentThemeID"] for _, r in theme_df.iterrows()}
theme_levels = {r["ThemeID"]: r["Level"] for _, r in theme_df.iterrows()}

coded_rows = []
for i in range(len(df)):
    item = assign_map.get(i, {"assignments": [], "rationale": ""})
    assigns = item.get("assignments", [])
    assigns = sorted(assigns, key=lambda a: a.get("confidence", 0.0), reverse=True)
    assigns = assigns[: (max_codes if allow_multicode else 1)]

    codes = [a.get("theme_id") for a in assigns]
    confs = [float(a.get("confidence", 0.0)) for a in assigns]
    # Map each sub-theme to its corresponding major theme label (aligned with codes)
    major_ids_aligned = [(parent_map.get(code_id) or code_id) for code_id in codes]
    major_labels_aligned = [label_map.get(mid, "") for mid in major_ids_aligned]

    # Determine primary theme for single code view
    primary_theme_id = codes[0] if codes else ""
    primary_major = major_map.get(primary_theme_id, "")
    primary_sub = primary_theme_id if theme_levels.get(primary_theme_id) == "Sub" else ""

    code_labels = [label_map.get(code_id, "") for code_id in codes if code_id]
    codes_str = "; ".join([c for c in code_labels if c])

    top_confidence = float(max(confs)) if confs else 0.0

    row = {
        "RowIndex": i,
        f"{question_label}_OE": ser.iloc[i],
        f"{question_label}_MajorTheme1": major_labels_aligned[0] if len(major_labels_aligned) > 0 else "",
        f"{question_label}_MajorTheme2": major_labels_aligned[1] if len(major_labels_aligned) > 1 else "",
        f"{question_label}_MajorTheme3": major_labels_aligned[2] if len(major_labels_aligned) > 2 else "",
        f"{question_label}_MajorTheme1_confidence": confs[0] if len(confs) > 0 else np.nan,
        f"{question_label}_MajorTheme2_confidence": confs[1] if len(confs) > 1 else np.nan,
        f"{question_label}_MajorTheme3_confidence": confs[2] if len(confs) > 2 else np.nan,
        f"{question_label}_MinorTheme1": code_labels[0] if len(code_labels) > 0 else "",
        f"{question_label}_MinorTheme2": code_labels[1] if len(code_labels) > 1 else "",
        f"{question_label}_MinorTheme3": code_labels[2] if len(code_labels) > 2 else "",
        f"{question_label}_MinorTheme1_confidence": confs[0] if len(confs) > 0 else np.nan,
        f"{question_label}_MinorTheme2_confidence": confs[1] if len(confs) > 1 else np.nan,
        f"{question_label}_MinorTheme3_confidence": confs[2] if len(confs) > 2 else np.nan,
        "IsMultiCoded": (len(codes) > 1),
    }

    # Carry through IDs
    for c in pass_id_cols:
        row[c] = df.loc[i, c]
    coded_rows.append(row)

coded_df = pd.DataFrame(coded_rows)

# Order columns
id_before = pass_id_cols.copy()
base_cols = [
    "RowIndex",
    f"{question_label}_OE",
    f"{question_label}_MajorTheme1",
    f"{question_label}_MajorTheme2",
    f"{question_label}_MajorTheme3",
    f"{question_label}_MajorTheme1_confidence",
    f"{question_label}_MajorTheme2_confidence",
    f"{question_label}_MajorTheme3_confidence",
    f"{question_label}_MinorTheme1",
    f"{question_label}_MinorTheme2",
    f"{question_label}_MinorTheme3",
    f"{question_label}_MinorTheme1_confidence",
    f"{question_label}_MinorTheme2_confidence",
    f"{question_label}_MinorTheme3_confidence",
    "IsMultiCoded",
]
ordered_cols = id_before + base_cols
coded_df = coded_df[ordered_cols]

# Review & Verification Section
st.divider()
st.subheader("Review & Verification")

# Show coded data preview
st.write("**Coded Data Preview:**")
st.dataframe(coded_df.head(20), width="stretch")

# Identify responses needing review
low = float(low_thresh)
flagged = []
for item in st.session_state["assigned_raw"]:
    confs = [a.get("confidence", 0.0) for a in item.get("assignments", [])]
    top_conf = max(confs) if confs else 0.0
    decision = item.get("decision")
    is_skipped_empty = (not item.get("assignments")) and str(item.get("rationale", "")).lower().startswith("skipped: empty")
    needs_review = (not is_skipped_empty) and (decision == "needs_review" or top_conf < low)
    if needs_review:
        idx = item.get("idx")
        item_for_review = dict(item)
        item_for_review["text"] = ser_ai.iloc[idx] if isinstance(idx, int) and 0 <= idx < len(ser_ai) else ""
        flagged.append(item_for_review)

# Needs review table
if flagged:
    review_rows = []
    for item in flagged:
        idx = item.get("idx")
        assigns = item.get("assignments", [])
        assigns = sorted(assigns, key=lambda a: a.get("confidence", 0.0), reverse=True)
        code_ids = [a.get("theme_id", "") for a in assigns if a.get("theme_id")]
        code_labels = [label_map.get(cid, cid) for cid in code_ids]
        conf_val = item.get("confidence")
        if not isinstance(conf_val, (int, float)):
            conf_val = max([a.get("confidence", 0.0) for a in assigns] or [0.0])

        review_rows.append({
            "RowIndex": idx,
            "Response": item.get("text", ""),
            "AssignedCodes": "; ".join(code_labels),
            "Confidence": conf_val,
            "Decision": item.get("decision", "needs_review"),
            "Rationale": item.get("rationale", ""),
        })

    needs_review_df = pd.DataFrame(review_rows)
    st.write("**Needs review**")
    st.dataframe(needs_review_df, width="stretch")

    csv_buf = needs_review_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download needs_review CSV",
        data=csv_buf,
        file_name=f"needs_review_{today_stamp()}.csv",
        mime="text/csv",
    )

# Review options (automatic verification only)
if flagged:
    st.write(f"**⚠️ {len(flagged)} responses flagged for review (confidence < {low_thresh})**")

    # Analyze low-confidence patterns for theme suggestions
    pattern_analysis = analyze_low_confidence_patterns(flagged, st.session_state.get("question_context"))

    if pattern_analysis["patterns"]:
        with st.expander("🔍 **Theme Pattern Analysis** - Potential New Themes", expanded=False):
            st.write("**Detected patterns in low-confidence responses:**")

            for pattern in pattern_analysis["patterns"]:
                st.write(f"**{pattern['pattern_name']}** ({pattern['response_count']} responses)")
                st.caption(f"Common words: {', '.join(pattern['common_words'])}")

                # Show sample responses
                with st.expander(f"Sample responses for {pattern['pattern_name']}", expanded=False):
                    for i, sample in enumerate(pattern['sample_responses'], 1):
                        st.write(f"{i}. {sample}")

            # Add new theme functionality
            st.write("**💡 Add New Theme/Sub-theme:**")

            col1, col2 = st.columns(2)
            with col1:
                new_theme_type = st.selectbox("Theme Type", ["sub_theme", "major_theme"], key="new_theme_type")

            with col2:
                if new_theme_type == "sub_theme":
                    # Get available major themes
                    major_theme_options = []
                    for major in st.session_state["theme_dict"].get("major_themes", []):
                        major_theme_options.append((major["id"], major["label"]))

                    parent_theme = st.selectbox("Parent Major Theme", major_theme_options, key="parent_theme")
                else:
                    parent_theme = None

            new_theme_name = st.text_input("New Theme Name", key="new_theme_name")
            new_theme_definition = st.text_area("Theme Definition", key="new_theme_definition")

            col3, col4 = st.columns(2)
            with col3:
                if st.button("➕ Add New Theme", type="secondary"):
                    if new_theme_name and new_theme_definition:
                        new_theme = {
                            "type": new_theme_type,
                            "theme_name": new_theme_name,
                            "definition": new_theme_definition,
                            "parent_theme_id": parent_theme[0] if parent_theme else None,
                            "sample_responses": [pattern["sample_responses"][0] for pattern in pattern_analysis["patterns"]]
                        }

                        # Add to theme dictionary
                        updated_theme_dict = add_new_theme_to_dictionary(st.session_state["theme_dict"], new_theme)
                        st.session_state["theme_dict"] = updated_theme_dict

                        st.success(f"✅ Added new {new_theme_type}: {new_theme_name}")

                        # Offer to re-assign with new themes
                        if st.button("🔄 Re-assign All Responses with New Themes", type="primary"):
                            pass
                            # This would trigger a re-assignment - for now just show a message
                            st.success("Theme dictionary updated! Consider re-running the assignment process to use the new themes.")
                        st.rerun()
                    else:
                        st.error("Please provide both theme name and definition")

            with col4:
                if st.button("🤖 AI Suggest Themes", type="secondary"):
                    pass

    if not st.session_state.get("_auto_verified"):
        with st.spinner("Auto re-checking low confidence assignments..."):
            verified, usage_verify = verify_low_confidence(
                client,
                model,
                st.session_state["theme_dict"],
                flagged,
                low_thresh=low,
                max_codes=(max_codes if allow_multicode else 1),
                seed=seed,
            )

            # Replace items by idx - ensure clean structure
            by_idx = {x["idx"]: x for x in st.session_state["assigned_raw"]}
            for v in verified:
                assigns = v.get("assignments", []) or []
                confs = [a.get("confidence", 0.0) for a in assigns]
                top_conf = max(confs) if confs else 0.0
                decision = "needs_review" if (top_conf < low or top_conf <= 0.4) else "ok"
                cleaned_item = {
                    "idx": v["idx"],
                    "assignments": assigns,
                    "subtheme_ids": [a.get("theme_id") for a in assigns if a.get("theme_id")],
                    "confidence": top_conf,
                    "decision": decision,
                    "rationale": "Auto re-check applied",
                }
                by_idx[v["idx"]] = cleaned_item
            st.session_state["assigned_raw"] = [by_idx[i] for i in sorted(by_idx.keys())]
            st.session_state["_auto_verified"] = True

            st.success(f"✅ Auto re-checked {len(verified)} assignments")
    else:
        st.caption("Auto-verification already completed for this run.")
else:
    st.success("✅ All responses have high confidence scores - no review needed!")


# Theme distribution for charting, with volume weights
st.divider()
st.subheader("Theme distribution")

# Compute support counts using primary MinorTheme1 as assignment for counting
count_series = coded_df[f"{question_label}_MinorTheme1"].replace("", np.nan).dropna()
counts = count_series.value_counts().rename_axis("Theme").reset_index(name="Count")

# Attach Major label for grouping
rev_label_to_id = {v: k for k, v in label_map.items()}
counts["ThemeID"] = counts["Theme"].map(rev_label_to_id)
counts["MajorID"] = counts["ThemeID"].map(lambda x: parent_map.get(x, x))
counts["MajorLabel"] = counts["MajorID"].map(label_map)

# Skip the table - go straight to Advanced Analytics

# Advanced Analytics
st.divider()
st.subheader("Advanced Analytics")

# Statistical summary - moved to top
st.write("**Statistical Summary**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_responses = len(coded_df)
    st.metric("Total Responses", f"{total_responses:,}")

with col2:
    coded_responses = len(coded_df[coded_df[f"{question_label}_MinorTheme1"] != ""])
    st.metric("Coded Responses", f"{coded_responses:,}")

with col3:
    coding_rate = (coded_responses / total_responses * 100) if total_responses > 0 else 0
    st.metric("Coding Rate", f"{coding_rate:.1f}%")

with col4:
    avg_confidence = coded_df[f"{question_label}_MinorTheme1_confidence"].mean()
    st.metric("Avg Confidence", f"{avg_confidence:.2f}")

# Theme distribution chart with interactive legend
st.write("**Theme Distribution**")

# Prepare data for charting
major_counts = coded_df[f"{question_label}_MajorTheme1"].value_counts()
sub_counts = coded_df[f"{question_label}_MinorTheme1"].value_counts()

# Create dataframes
major_df = pd.DataFrame({
    'Theme': major_counts.index,
    'Count': major_counts.values,
    'Level': 'Major'
}) if not major_counts.empty else pd.DataFrame()

sub_df = pd.DataFrame({
    'Theme': sub_counts.index,
    'Count': sub_counts.values,
    'Level': 'Sub'
}) if not sub_counts.empty else pd.DataFrame()

# Combine all theme data for visualization
chart_data = pd.DataFrame()

if not major_df.empty:
    chart_data = pd.concat([chart_data, major_df], ignore_index=True)

if not sub_df.empty:
    chart_data = pd.concat([chart_data, sub_df], ignore_index=True)

if not chart_data.empty:
    # Create hierarchical sorting: Major themes first (descending), then sub-themes under each major theme
    # First, we need to get the major theme for each sub-theme
    major_theme_map = {}
    for _, row in chart_data.iterrows():
        if row['Level'] == 'Major':
            major_theme_map[row['Theme']] = row['Theme']
        else:
            # For sub-themes, we need to find which major theme they belong to
            # This requires looking at the theme dictionary structure
            for major_theme in st.session_state.get("theme_dict", {}).get("major_themes", []):
                if major_theme.get("label") == row['Theme']:
                    major_theme_map[row['Theme']] = major_theme.get("label")
                    break
                for sub_theme in major_theme.get("subs", []):
                    if sub_theme.get("label") == row['Theme']:
                        major_theme_map[row['Theme']] = major_theme.get("label")
                        break
    
    # Add major theme column for sorting
    chart_data['MajorTheme'] = chart_data['Theme'].map(major_theme_map)
    chart_data['MajorTheme'] = chart_data['MajorTheme'].fillna(chart_data['Theme'])
    
    # Filter out blank/empty themes
    chart_data = chart_data[
        (chart_data['Theme'].notna()) & 
        (chart_data['Theme'].str.strip() != '') & 
        (chart_data['Theme'] != 'nan')
    ].copy()
    
    # Create hierarchical ordering: Major themes by count (descending), then sub-themes under each major theme by count (descending)
    ordered_themes = []
    
    # Get major themes sorted by count (descending)
    major_themes = chart_data[chart_data['Level'] == 'Major'].sort_values('Count', ascending=False)
    
    for _, major_row in major_themes.iterrows():
        major_theme = major_row['Theme']
        ordered_themes.append(major_theme)
        
        # Get sub-themes for this major theme, sorted by count (descending)
        sub_themes = chart_data[
            (chart_data['Level'] == 'Sub') & 
            (chart_data['MajorTheme'] == major_theme)
        ].sort_values('Count', ascending=False)
        
        for _, sub_row in sub_themes.iterrows():
            ordered_themes.append(sub_row['Theme'])
    
    # Reorder chart_data to match the ordered_themes (keep original order, don't reverse)
    # Handle duplicates by using a unique index
    chart_data = chart_data.reset_index(drop=True)
    chart_data['Theme_Index'] = chart_data.index
    
    # Create a mapping from theme to the ordered position (don't reverse here)
    theme_to_position = {theme: i for i, theme in enumerate(ordered_themes)}
    chart_data['Order'] = chart_data['Theme'].map(theme_to_position)
    
    # Sort by the order and drop the helper columns
    chart_data = chart_data.sort_values('Order').drop(['Theme_Index', 'Order'], axis=1).reset_index(drop=True)
    
    # Create interactive horizontal bar chart using plotly
    import plotly.express as px
    
    # Create color mapping
    color_map = {'Major': '#1976d2', 'Sub': '#42a5f5'}
    
    # Create horizontal bar chart with interactive legend
    fig = px.bar(
        chart_data, 
        x='Count', 
        y='Theme',
        color='Level',
        color_discrete_map=color_map,
        orientation='h',  # This makes it horizontal
        title="Theme Distribution",
        height=max(400, len(chart_data) * 30),  # Dynamic height
        hover_data={'Count': True, 'Level': True}
    )
    
    # Update layout for better readability
    fig.update_layout(
        showlegend=True,
        xaxis_title="Count",
        yaxis_title="",
        yaxis={'categoryorder': 'array', 'categoryarray': ordered_themes[::-1]},  # Reverse for top-to-bottom display
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update traces for better styling
    fig.update_traces(
        marker_line_width=0,
        hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Level: %{customdata[0]}<extra></extra>',
        customdata=chart_data[['Level']]
    )
    
    # Display the interactive chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the data table for reference
    st.write("**Theme Distribution Data:**")
    st.dataframe(chart_data, width="stretch")
else:
    st.info("No themes selected for display.")

# Theme distribution analysis using tiny_threshold
st.write("**Theme Distribution Analysis**")
theme_analysis = analyze_theme_distribution(coded_df, tiny_threshold, st.session_state.get("theme_dict"))
pass
if dynamic_thresholds.get("recommendations"):
    for rec in dynamic_thresholds["recommendations"]:
        st.caption(f"• {rec}")

# ------------------------------
# Export
# ------------------------------

st.divider()
st.subheader("Export")

file_name = f"{question_label}_CODED.xlsx"

# Build Theme Dictionary with shares
major_support = counts.groupby("MajorLabel")["Count"].sum().rename("MajorCount").reset_index()
sub_df = counts.merge(major_support, on="MajorLabel", how="left")
sub_df["SharePercent"] = (sub_df["Count"] / max(1, len(df))) * 100

theme_export = theme_df.copy()
# Fill support from counts where available
support_map = sub_df.set_index("Theme")["Count"].to_dict()
share_map = sub_df.set_index("Theme")["SharePercent"].to_dict()

theme_export["SupportCount"] = theme_export["Label"].map(lambda x: support_map.get(x, 0))
theme_export["SharePercent"] = theme_export["Label"].map(lambda x: round(share_map.get(x, 0.0), 2))

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
    coded_df.to_excel(writer, sheet_name="Coded Data", index=False)
    theme_export.to_excel(writer, sheet_name="Theme Dictionary", index=False)
    question_map_df = pd.DataFrame([{"QuestionLabel": question_label, "QuestionText": text_col, "RunId": st.session_state.get("run_id", "")}])
    question_map_df.to_excel(writer, sheet_name="Question Map", index=False)
    governance_log = st.session_state.get("theme_governance_log") or []
    if governance_log:
        governance_df = governance_log_to_df(governance_log)
        governance_df.to_excel(writer, sheet_name="Theme Governance Log", index=False)

st.download_button("Download XLSX", data=buf.getvalue(), file_name=file_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------
# Run report
# ------------------------------

st.divider()
st.subheader("Run report")

cache_stats = get_cache_stats()
cache_stats_dict = {
    "hits": cache_stats.hits,
    "misses": cache_stats.misses,
    "saved_prompt_tokens": cache_stats.saved_prompt_tokens,
    "saved_completion_tokens": cache_stats.saved_completion_tokens,
}

theme_params = {
    "model": model,
    "seed": seed,
    "prompt_versions": prompt_versions,
}
assignment_params = run_settings.copy()
assignment_params["guidance_text"] = active_guidance
theme_params["guidance_text"] = active_guidance

run_audit = build_run_audit(
    run_id=st.session_state.get("run_id", ""),
    dataset_sig=dataset_sig,
    question_text=text_col,
    theme_params=theme_params,
    governance_log=st.session_state.get("theme_governance_log", []),
    assignment_params=assignment_params,
    assignments=st.session_state.get("assigned_raw", []),
    cache_stats=cache_stats_dict,
    usage_totals=st.session_state.get("_usage_totals", {}),
)

run_report_json = json.dumps(run_audit, indent=2, ensure_ascii=False)
st.download_button(
    "Download run report JSON",
    data=run_report_json,
    file_name=f"run_report_{st.session_state.get('run_id', '')}.json",
    mime="application/json",
)

# ------------------------------
# Cost summary
# ------------------------------

# Initialize usage tracking
if "_usage_totals" not in st.session_state:
    st.session_state["_usage_totals"] = {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}

# Accumulate usage from completed steps
total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}

# Add usage from theme generation if available
if "usage_theme" in locals():
    total_usage["prompt_tokens"] += usage_theme.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += usage_theme.get("completion_tokens", 0)
    total_usage["embedding_tokens"] += usage_theme.get("embedding_tokens", 0)

# Add usage from assignment if available
if "usage_assign" in locals():
    total_usage["prompt_tokens"] += usage_assign.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += usage_assign.get("completion_tokens", 0)
    total_usage["embedding_tokens"] += usage_assign.get("embedding_tokens", 0)

# Add usage from verification if available
if "usage_verify" in locals():
    total_usage["prompt_tokens"] += usage_verify.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += usage_verify.get("completion_tokens", 0)
    total_usage["embedding_tokens"] += usage_verify.get("embedding_tokens", 0)

# Update session state
st.session_state["_usage_totals"] = total_usage

# Pricing table (as of 2024)
pricing_table = {
    "gpt-5": {"prompt_per_1k": 0.005, "completion_per_1k": 0.015},  # Estimated pricing
    "gpt-4o": {"prompt_per_1k": 0.005, "completion_per_1k": 0.015},
    "gpt-4o-mini": {"prompt_per_1k": 0.0005, "completion_per_1k": 0.0015},
    "gpt-4-turbo": {"prompt_per_1k": 0.01, "completion_per_1k": 0.03},
}

if total_usage["prompt_tokens"] > 0 or total_usage["completion_tokens"] > 0:
    # Calculate cost for single model (GPT-5)
    pricing = pricing_table.get(model, {"prompt_per_1k": 0.005, "completion_per_1k": 0.015})
    estimated_cost = fmt_cost(total_usage["prompt_tokens"], total_usage["completion_tokens"], pricing)
    
    st.divider()
    st.subheader("Cost Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prompt Tokens", f"{total_usage['prompt_tokens']:,}")
    with col2:
        st.metric("Completion Tokens", f"{total_usage['completion_tokens']:,}")
    with col3:
        st.metric("Embedding Tokens", f"{total_usage['embedding_tokens']:,}")
    with col4:
        st.metric("Total Cost (GPT-5)", f"${estimated_cost:.4f}")
    
    total_tokens_all = total_usage["prompt_tokens"] + total_usage["completion_tokens"] + total_usage["embedding_tokens"]
    per_1k = (total_tokens_all / max(1, len(df))) * 1000
    st.caption(f"Tokens per 1k responses (all stages): {per_1k:.0f}")
    cache_stats = get_cache_stats()
    st.caption(
        f"Cache hits: {cache_stats.hits} | misses: {cache_stats.misses} | "
        f"saved tokens: {cache_stats.saved_prompt_tokens + cache_stats.saved_completion_tokens}"
    )
    pass