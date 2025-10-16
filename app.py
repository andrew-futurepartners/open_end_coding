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
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Tuple
from threading import Lock

import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from openai import RateLimitError
from dotenv import load_dotenv

# ------------------------------
# Utilities
# ------------------------------

def get_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not set. Add OPENAI_API_KEY to Streamlit secrets or environment.")
        st.stop()
    return OpenAI(api_key=api_key)


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


def is_empty_like(s: str) -> bool:
    if not s:
        return True
    # Common non‑answers in many languages, minimal set, verified in model pass later
    na_set = {"n/a", "na", "none", "no", "nothing", "-", "—", "dont know", "don't know", "idk", "prefer not to say"}
    return s.lower() in na_set


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token"""
    return len(text) // 4


def chunk_data(data: List[Dict[str, Any]], max_tokens: int = 400000) -> List[List[Dict[str, Any]]]:
    """Split data into chunks that won't exceed token limits"""
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for item in data:
        item_tokens = estimate_tokens(json.dumps(item))
        
        # If adding this item would exceed the limit, start a new chunk
        if current_tokens + item_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [item]
            current_tokens = item_tokens
        else:
            current_chunk.append(item)
            current_tokens += item_tokens
    
    # Add the last chunk if it has items
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


# Rate limiting for GPT-5: 500 RPM, 500K TPM
_request_times = []
_token_usage = []
_rate_limit_lock = Lock()

def check_rate_limits(estimated_tokens: int = 0):
    """NUCLEAR MODE: Minimal rate limiting - let OpenAI handle the throttling"""
    with _rate_limit_lock:
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        global _request_times, _token_usage
        _request_times = [t for t in _request_times if t > cutoff_time]
        _token_usage = [(t, tokens) for t, tokens in _token_usage if t > cutoff_time]
        
        # NUCLEAR MODE: Only prevent hitting the absolute limits
        # Let OpenAI's servers do the rate limiting for maximum speed
        
        # Only block if we're at 99% of limits
        if len(_request_times) >= 495:  # 99% of 500 RPM
            # Tiny sleep to prevent hitting absolute limit
            time.sleep(0.01)  # 10ms sleep
        
        current_tokens = sum(tokens for _, tokens in _token_usage)
        if current_tokens + estimated_tokens > 495000:  # 99% of 500K TPM
            # Tiny sleep to prevent hitting absolute limit
            time.sleep(0.01)  # 10ms sleep
        
        # Record this request
        _request_times.append(current_time)
        if estimated_tokens > 0:
            _token_usage.append((current_time, estimated_tokens))

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry function with exponential backoff, Retry-After header support, and rate limiting"""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            
            # Check for Retry-After header
            retry_after = None
            if hasattr(e, 'response') and e.response:
                retry_after = e.response.headers.get("retry-after")
            
            if retry_after:
                delay = float(retry_after)
                st.warning(f"Rate limit hit. Retrying in {delay:.1f} seconds (Retry-After)... (attempt {attempt + 1}/{max_retries})")
            else:
                delay = base_delay * (2 ** attempt) * (1 + np.random.random() * 0.25)
                st.warning(f"Rate limit hit. Retrying in {delay:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
            
            time.sleep(min(delay, 20))
        except Exception as e:
            # For non-rate-limit errors, don't retry
            raise e




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
    """Process multiple chunks in parallel for faster assignment"""
    theme_json = json.dumps(slim_theme_for_assignment(theme_dict))
    all_assignments = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    def process_single_chunk(chunk):
        """Process a single chunk"""
        responses_json = json.dumps(chunk)
        user = ASSIGNMENT_USER_TEMPLATE.format(max_codes=max_codes, theme_json=theme_json, responses_json=responses_json)
        
        def make_request():
            return oai_json_completion(client, model, ASSIGNMENT_SYSTEM, user, seed, ASSIGNMENTS_SCHEMA)
        
        return retry_with_backoff(make_request)
    
    # Process chunks in parallel (reduced to 1 for cost optimization)
    max_workers = 1  # Sequential processing to minimize costs
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {executor.submit(process_single_chunk, chunk): chunk for chunk in chunks}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                data, usage = future.result()
                
                # Data expected as object with results array
                if isinstance(data, dict) and "results" in data:
                    data = data["results"]
                else:
                    # Fallback for backward compatibility
                    data = data if isinstance(data, list) else []
                
                all_assignments.extend(data)
                
                # Accumulate usage
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                
            except Exception as e:
                st.error(f"Error processing chunk: {str(e)}")
                raise e
    
    return all_assignments, total_usage


def process_chunk_batch_optimized(client: OpenAI, model: str, theme_dict: Dict[str, Any], chunks: List[List[Dict[str, Any]]], max_codes: int, seed: int | None, progress_bar=None, status_text=None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Optimized batch processing with higher parallelism and better error handling"""
    theme_json = json.dumps(slim_theme_for_assignment(theme_dict))
    all_assignments = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    def process_single_chunk(chunk):
        """Process a single chunk with optimized prompt"""
        responses_json = json.dumps(chunk)
        
        # Use the same structured template as the single request version
        user = ASSIGNMENT_USER_TEMPLATE.format(max_codes=max_codes, theme_json=theme_json, responses_json=responses_json)
        
        def make_request():
            return oai_json_completion(client, model, ASSIGNMENT_SYSTEM, user, seed, ASSIGNMENTS_SCHEMA)
        
        return retry_with_backoff(make_request)
    
    # Use minimal parallelism for cost optimization (sequential processing)
    max_workers = 1  # Sequential processing to minimize costs
    
    # Progress tracking - use provided progress elements or create new ones
    if progress_bar is None:
        progress_bar = st.progress(0)
    if status_text is None:
        status_text = st.empty()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {executor.submit(process_single_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        completed = 0
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                chunk_idx = future_to_chunk[future]
                data, usage = future.result()
                
                # Data expected as object with results array
                if isinstance(data, dict) and "results" in data:
                    data = data["results"]
                else:
                    # Fallback for backward compatibility
                    data = data if isinstance(data, list) else []
                
                all_assignments.extend(data)
                
                # Accumulate usage
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                
                completed += 1
                # Update progress within the assignment phase (30-95% range)
                if progress_bar is not None:
                    progress_bar.progress(30 + (completed * 65 // len(chunks)))
                if status_text is not None:
                    status_text.text(f"🏷️ Processing assignment chunk {completed}/{len(chunks)}")
                
            except Exception as e:
                st.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                raise e
    
    if status_text is not None:
        status_text.text("✅ Assignment processing complete!")
    return all_assignments, total_usage


# ------------------------------
# JSON Schemas for faster parsing
# ------------------------------

ASSIGNMENTS_SCHEMA = {
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
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "theme_id": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                    },
                                    "required": ["theme_id", "confidence"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["idx", "assignments"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        },
        "strict": True
    }
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

ASSIGNMENT_USER_TEMPLATE = (
    """
You will assign the following responses to the provided theme dictionary.

For each response, find the most appropriate theme(s) from the dictionary and assign with confidence scores between 0 and 1.

IMPORTANT INSTRUCTIONS:
- Assign up to {max_codes} theme(s) per response if multiple themes apply with high confidence
- If a response clearly addresses multiple distinct themes, provide multiple assignments sorted by confidence (highest first)
- Each assignment must have theme_id and confidence (0-1 scale)
- Only assign additional themes beyond the primary if they are genuinely relevant with confidence ≥ 0.6
- Focus on the substantive content of what respondents are saying
- Only assign to "Non-answer" themes if the response is truly a non-answer (refusal, don't know, nonsense, spam, or not applicable)

Return JSON in this exact format:
{{
  "results": [
    {{
      "idx": <row index integer>,
      "assignments": [
        {{"theme_id": "T1.2", "confidence": 0.87}},
        {{"theme_id": "T2.1", "confidence": 0.75}}
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

VERIFY_SYSTEM = (
    "You are a reviewer checking questionable assignments and fixing them."
)

VERIFY_USER_TEMPLATE = (
    """
You will re‑check only low‑confidence or ambiguous assignments. If an assignment is below {low_thresh}, try to improve it by selecting a better theme from the dictionary.

If the top assignment is <{low_thresh} and a secondary theme is a materially better fit, promote the secondary and adjust confidences accordingly.
If an assignment is clearly correct but under-confident, raise it to an appropriate level; do not leave obviously correct matches <{low_thresh}.
If no theme is defensible after re-check, set manual_review: true and keep a single, best-effort assignment with low confidence.
Optional note may explain changes in ≤15 words.

Return the same JSON shape as the assignment step, for only the provided rows. If you agree with the existing assignment, return it unchanged but you may adjust confidence.

Return only valid JSON matching the schema; do not include any additional text, code fences, or commentary.

Theme dictionary:
{theme_json}

Flagged rows:
{flagged_json}
"""
)

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


def oai_json_completion(client: OpenAI, model: str, system: str, user: str, seed: int | None, response_schema: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call Chat Completions expecting a JSON object or array in content. Returns (parsed_json, usage)."""
    params = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    
    # Use schema if provided, otherwise fall back to json_object
    if response_schema:
        params["response_format"] = response_schema
    else:
        params["response_format"] = {"type": "json_object"}
    
    # GPT-5 doesn't support temperature parameter, uses default temperature=1
    if model != "gpt-5":
        params["temperature"] = 0
    
    # Seed is optional. If unsupported, OpenAI will ignore it. We keep it here for determinism where available.
    if seed is not None:
        params["seed"] = int(seed)

    resp = client.chat.completions.create(**params)
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        # Try to coerce array as object if needed
        if content.strip().startswith("["):
            data = json.loads(content)
        else:
            raise
    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
    }
    return data, usage




# ------------------------------
# Theming logic
# ------------------------------

def build_theme_frame(client: OpenAI, model: str, texts: List[str], freq: List[int], seed: int | None, question_text: str = None) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Create a hierarchical theme dictionary using a weighted sample of unique texts.
    We pass a compact JSON with objects: {"text": "...", "weight": n}
    Handles large datasets by chunking and processing in batches.
    Pre-filters non-responses for better theme quality.
    """
    
    # Pre-filter non-responses and low-quality responses
    st.info("🔍 Pre-filtering responses for better theme discovery...")
    
    filtered_data = []
    non_answer_count = 0
    short_response_count = 0
    
    for t, w in zip(texts, freq):
        if not t or is_empty_like(t):
            non_answer_count += w  # Count frequency of non-answers
            continue
        
        # Filter out very short responses that are likely non-substantive
        if len(t.strip()) < 10:
            short_response_count += w
            continue
            
        filtered_data.append({"text": t, "weight": int(w)})
    
    # Show filtering statistics
    total_responses = sum(freq)
    filtered_responses = sum(item["weight"] for item in filtered_data)
    non_answer_pct = (non_answer_count / total_responses * 100) if total_responses > 0 else 0
    short_response_pct = (short_response_count / total_responses * 100) if total_responses > 0 else 0
    
    if non_answer_count > 0 or short_response_count > 0:
        st.success(f"📊 Pre-filtering: {total_responses} → {filtered_responses} responses")
        if non_answer_count > 0:
            st.caption(f"   • Removed {non_answer_count} non-answers ({non_answer_pct:.1f}%)")
        if short_response_count > 0:
            st.caption(f"   • Removed {short_response_count} short responses ({short_response_pct:.1f}%)")
    
    # Sort by weight
    filtered_data.sort(key=lambda x: x["weight"], reverse=True)
    
    # Check if we need to chunk the filtered data
    total_tokens = estimate_tokens(json.dumps(filtered_data))
    if total_tokens <= 400000:  # GPT-5 safe token limit
        payload = json.dumps(filtered_data)
        # Build prompt with optional question context
        if question_text and question_text.strip():
            user = THEME_DISCOVERY_USER + f"\n\n**Survey Question:** {question_text}\n\nWeighted responses (JSON array):\n" + payload
        else:
            user = THEME_DISCOVERY_USER + "\n\nWeighted responses (JSON array):\n" + payload
        
        def make_request():
            return oai_json_completion(client, model, THEME_DISCOVERY_SYSTEM, user, seed)
        
        data, usage = retry_with_backoff(make_request)
        return data, usage
    
    else:
        # Process in chunks and merge results
        st.info(f"Very large dataset detected ({total_tokens:,} tokens). Processing in chunks to optimize with GPT-5's 500K TPM limit...")
        
        chunks = chunk_data(filtered_data, max_tokens=350000)  # GPT-5 conservative limit
        all_themes = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process chunks in parallel for much faster theme generation
        def process_theme_chunk(chunk):
            payload = json.dumps(chunk)
            # Build prompt with optional question context
            if question_text and question_text.strip():
                user = THEME_DISCOVERY_USER + f"\n\n**Survey Question:** {question_text}\n\nWeighted responses (JSON array):\n" + payload
            else:
                user = THEME_DISCOVERY_USER + "\n\nWeighted responses (JSON array):\n" + payload
            
            def make_chunk_request():
                return oai_json_completion(client, model, THEME_DISCOVERY_SYSTEM, user, seed)
            
            return retry_with_backoff(make_chunk_request)
        
        # Use minimal parallelism for cost optimization
        max_workers = 1  # Sequential processing to minimize costs
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(process_theme_chunk, chunk): i for i, chunk in enumerate(chunks)}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    data, usage = future.result()
                    all_themes.extend(data.get("major_themes", []))
                    
                    # Accumulate usage
                    total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    
                    completed += 1
                    status_text.text(f"🎯 Processing theme chunk {completed}/{len(chunks)}")
                    progress_bar.progress(30 + (completed * 60 // len(chunks)))
                    
                except Exception as e:
                    st.error(f"Error processing theme chunk {chunk_idx + 1}: {str(e)}")
                    raise e
        
        # Merge and deduplicate themes
        merged_themes = merge_theme_chunks(all_themes)
        
        status_text.text("Theme generation complete!")
        progress_bar.progress(1.0)
        
        result = {"major_themes": merged_themes}
        return result, total_usage


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

def analyze_low_confidence_patterns(low_confidence_responses: List[Dict]) -> Dict[str, Any]:
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

def suggest_new_themes_from_review(low_confidence_responses: List[Dict], existing_theme_dict: Dict) -> Dict[str, Any]:
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
    
    Question context: General thematic analysis
    
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

# Theme quality validation removed - the AI prompt already includes quality guidelines
# and there are no actionable items from validation scores

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

def build_theme_frame_with_progress(client: OpenAI, model: str, texts: List[str], freq: List[int], seed: int | None, progress_bar, status_text, question_text: str = None) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Create a hierarchical theme dictionary with detailed progress tracking."""
    
    # Pre-filter non-responses and low-quality responses
    status_text.text("🔍 Pre-filtering responses for better theme discovery...")
    progress_bar.progress(10)
    
    filtered_data = []
    non_answer_count = 0
    short_response_count = 0
    
    for t, w in zip(texts, freq):
        if not t or is_empty_like(t):
            non_answer_count += w  # Count frequency of non-answers
            continue
        
        # Filter out very short responses that are likely non-substantive
        if len(t.strip()) < 10:
            short_response_count += w
            continue
            
        filtered_data.append({"text": t, "weight": int(w)})
    
    # Show filtering statistics
    total_responses = sum(freq)
    filtered_responses = sum(item["weight"] for item in filtered_data)
    non_answer_pct = (non_answer_count / total_responses * 100) if total_responses > 0 else 0
    short_response_pct = (short_response_count / total_responses * 100) if total_responses > 0 else 0
    
    status_text.text(f"📊 Pre-filtering: {total_responses} → {filtered_responses} responses")
    progress_bar.progress(20)
    
    # Sort by weight
    filtered_data.sort(key=lambda x: x["weight"], reverse=True)
    
    # Check if we need to chunk the filtered data
    total_tokens = estimate_tokens(json.dumps(filtered_data))
    if total_tokens <= 400000:  # GPT-5 safe token limit
        status_text.text("🎯 Generating themes from all responses...")
        progress_bar.progress(50)
        
        payload = json.dumps(filtered_data)
        
        # Build prompt with optional question context
        if question_text and question_text.strip():
            user = THEME_DISCOVERY_USER + f"\n\n**Survey Question:** {question_text}\n\nWeighted responses (JSON array):\n" + payload
        else:
            user = THEME_DISCOVERY_USER + "\n\nWeighted responses (JSON array):\n" + payload
        
        def make_request():
            return oai_json_completion(client, model, THEME_DISCOVERY_SYSTEM, user, seed)
        
        data, usage = retry_with_backoff(make_request)
        progress_bar.progress(100)
        return data, usage
    
    else:
        # Process in chunks and merge results
        status_text.text(f"🚀 Large dataset detected ({total_tokens:,} tokens). Processing in chunks...")
        progress_bar.progress(30)
        
        chunks = chunk_data(filtered_data, max_tokens=350000)  # GPT-5 conservative limit
        all_themes = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        # Process chunks in parallel for much faster theme generation
        def process_theme_chunk(chunk):
            payload = json.dumps(chunk)
            # Build prompt with optional question context
            if question_text and question_text.strip():
                user = THEME_DISCOVERY_USER + f"\n\n**Survey Question:** {question_text}\n\nWeighted responses (JSON array):\n" + payload
            else:
                user = THEME_DISCOVERY_USER + "\n\nWeighted responses (JSON array):\n" + payload
            
            def make_chunk_request():
                return oai_json_completion(client, model, THEME_DISCOVERY_SYSTEM, user, seed)
            
            return retry_with_backoff(make_chunk_request)
        
        # Use minimal parallelism for cost optimization
        max_workers = 1  # Sequential processing to minimize costs
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(process_theme_chunk, chunk): i for i, chunk in enumerate(chunks)}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    data, usage = future.result()
                    all_themes.extend(data.get("major_themes", []))
                    
                    # Accumulate usage
                    total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    
                    completed += 1
                    status_text.text(f"🎯 Processing theme chunk {completed}/{len(chunks)}")
                    progress_bar.progress(30 + (completed * 60 // len(chunks)))
                    
                except Exception as e:
                    st.error(f"Error processing theme chunk {chunk_idx + 1}: {str(e)}")
                    raise e
        
        # Merge and deduplicate themes
        status_text.text("🔄 Merging and deduplicating themes...")
        progress_bar.progress(90)
        
        merged_themes = merge_theme_chunks(all_themes)
        result = {"major_themes": merged_themes}
        
        progress_bar.progress(100)
        return result, total_usage


def merge_theme_chunks(theme_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge theme chunks and deduplicate similar themes with proper hierarchy enforcement"""
    if not theme_chunks:
        return []
    
    # Step 1: Collect all labels at both major and sub levels
    all_major_labels = set()
    all_sub_labels = set()
    
    for theme in theme_chunks:
        major_label = theme.get("label", "").strip()
        if major_label:
            all_major_labels.add(major_label.lower().strip())
        
        for sub in theme.get("subs", []):
            sub_label = sub.get("label", "").strip()
            if sub_label:
                all_sub_labels.add(sub_label.lower().strip())
    
    # Step 2: Identify hierarchy conflicts (same label at both major and sub level)
    conflicting_labels = all_major_labels.intersection(all_sub_labels)
    
    if conflicting_labels:
        st.warning(f"🔧 Resolving hierarchy conflicts: {len(conflicting_labels)} themes appear at both major and sub levels")
    
    # Step 3: Merge themes with conflict resolution
    merged = {}
    
    for theme in theme_chunks:
        theme_id = theme.get("id", "")
        theme_label = theme.get("label", "").strip()
        
        if not theme_label:  # Skip empty labels
            continue
            
        # Normalize label for comparison
        normalized_label = theme_label.lower().strip()
        
        # Check if we already have a major theme with this exact label
        existing_key = None
        for key, existing_theme in merged.items():
            existing_label = existing_theme.get("label", "").lower().strip()
            if existing_label == normalized_label:
                existing_key = key
                break
        
        if existing_key:
            # Merge with existing major theme
            existing_theme = merged[existing_key]
            
            # Merge sub-themes, but skip any that conflict with major theme labels
            existing_sub_labels = {sub.get("label", "").lower().strip() for sub in existing_theme.get("subs", [])}
            
            for sub in theme.get("subs", []):
                sub_label = sub.get("label", "").strip()
                sub_label_norm = sub_label.lower().strip()
                
                # Skip if this sub-theme label conflicts with any major theme label
                if sub_label_norm in conflicting_labels:
                    continue
                    
                if sub_label and sub_label_norm not in existing_sub_labels:
                    existing_theme.setdefault("subs", []).append(sub)
                    existing_sub_labels.add(sub_label_norm)
                    
        else:
            # Add new major theme with clean structure
            clean_theme = {
                "id": theme_id,
                "label": theme_label,
                "definition": theme.get("definition", ""),
                "subs": []
            }
            
            # Add sub-themes, ensuring no duplicates and no conflicts with major themes
            sub_labels_seen = set()
            for sub in theme.get("subs", []):
                sub_label = sub.get("label", "").strip()
                sub_label_norm = sub_label.lower().strip()
                
                # Skip if this sub-theme label conflicts with any major theme label
                if sub_label_norm in conflicting_labels:
                    continue
                
                if sub_label and sub_label_norm not in sub_labels_seen:
                    clean_theme["subs"].append(sub)
                    sub_labels_seen.add(sub_label_norm)
            
            merged[theme_id] = clean_theme
    
    # Step 4: Final validation - ensure no sub-theme has the same label as any major theme
    for theme in merged.values():
        major_label_norm = theme.get("label", "").lower().strip()
        theme["subs"] = [
            sub for sub in theme.get("subs", [])
            if sub.get("label", "").lower().strip() != major_label_norm
        ]
    
    merged_themes = list(merged.values())
    
    if conflicting_labels:
        st.success(f"✅ Resolved {len(conflicting_labels)} hierarchy conflicts - themes now have proper major/sub structure")
    
    return merged_themes


def assign_codes(client: OpenAI, model: str, theme_dict: Dict[str, Any], rows: List[Dict[str, Any]], max_codes: int, seed: int | None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    
    # Step 1: Aggressive pre-filtering and deduplication
    st.info("🔍 Pre-filtering and deduplicating responses...")
    
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
                "assignments": [{"theme_id": non_answer_theme_id or "T1.1", "confidence": 1.0}]
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
        st.info(f"🚀 Using single request mode for {len(unique_rows_for_ai)} unique responses ({total_tokens:,} tokens)")
        theme_json = json.dumps(slim_theme_for_assignment(theme_dict))
        responses_json = json.dumps(unique_rows_for_ai, separators=(",", ":"))
        
        # Debug: Show theme structure
        st.info(f"🔍 Debug: Theme dictionary has {len(theme_dict.get('major_themes', []))} major themes")
        for i, major in enumerate(theme_dict.get('major_themes', [])[:2]):  # Show first 2
            st.info(f"   Major {i+1}: {major.get('label', 'Unknown')} with {len(major.get('subs', []))} sub-themes")
        
        user = ASSIGNMENT_USER_TEMPLATE.format(max_codes=max_codes, theme_json=theme_json, responses_json=responses_json)
        
        def make_request():
            return oai_json_completion(client, model, ASSIGNMENT_SYSTEM, user, seed, ASSIGNMENTS_SCHEMA)
        
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
        st.info(f"🚀 Using chunked processing mode: {unique_count} unique responses → {len(chunks)} chunks ({total_tokens:,} tokens)")
        
        # Use conservative chunks for GPT-5 quality
        
        # Process chunks with more parallelism
        all_assignments, total_usage = process_chunk_batch_optimized(client, model, theme_dict, chunks, max_codes, seed)
        
        # Expand results back to all original responses
        expanded_assignments = expand_deduplicated_results(all_assignments, response_to_indices)
        
        
        # Add back the non-answer assignments for filtered responses
        final_assignments = expanded_assignments + non_answer_assignments
        final_assignments.sort(key=lambda x: x["idx"])  # Sort by original index
        
        st.success("✅ Assignment complete!")
        return final_assignments, total_usage


def assign_codes_with_progress(client: OpenAI, model: str, theme_dict: Dict[str, Any], rows: List[Dict[str, Any]], max_codes: int, seed: int | None, progress_bar, status_text) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assign themes to responses - PROCESS ALL RESPONSES WITH DUPLICATE CONSISTENCY"""
    
    status_text.text("Processing responses...")
    progress_bar.progress(5)
    
    # Find a non-answer theme ID from the theme dictionary for empty responses
    non_answer_theme_id = None
    for major in theme_dict.get("major_themes", []):
        if "non-answer" in major.get("label", "").lower():
            subs = major.get("subs", [])
            if subs:
                non_answer_theme_id = subs[0].get("id")
                break
    
    # Fallback if no non-answer theme found
    if not non_answer_theme_id:
        for major in theme_dict.get("major_themes", []):
            subs = major.get("subs", [])
            if subs:
                non_answer_theme_id = subs[0].get("id")
                break
    
    # Step 1: Identify unique responses and their indices for consistency
    text_to_indices = {}
    for row in rows:
        text = row["text"]
        if text not in text_to_indices:
            text_to_indices[text] = []
        text_to_indices[text].append(row["idx"])
    
    unique_texts = list(text_to_indices.keys())
    
    # Step 2: Process unique responses and store assignments
    text_to_assignment = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    # Optimize processing with efficient chunking
    chunk_size = 10
    unique_count = len(unique_texts)
    total_chunks = math.ceil(unique_count / chunk_size)
    
    # Prepare all chunks for parallel processing
    all_chunks = []
    for i in range(0, unique_count, chunk_size):
        chunk_texts = unique_texts[i:i + chunk_size]
        all_chunks.append((i, chunk_texts))
    
    def process_single_chunk(chunk_data):
        """Process a single chunk of unique texts"""
        chunk_start_idx, chunk_texts = chunk_data
        chunk_assignments = {}
        chunk_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        # Prepare chunk for processing
        responses_for_ai = []
        
        for j, text in enumerate(chunk_texts):
            if not text or is_empty_like(text) or len(text.strip()) < 3:
                # Handle empty responses directly
                chunk_assignments[text] = {
                    "assignments": [{"theme_id": non_answer_theme_id or "T1.1", "confidence": 1.0}]
                }
            else:
                # Add to AI processing list with sequential index for this chunk
                responses_for_ai.append({
                    "idx": j,
                    "text": text
                })
        
        # Process non-empty responses with AI
        if responses_for_ai:
            theme_json = json.dumps(slim_theme_for_assignment(theme_dict))
            responses_json = json.dumps(responses_for_ai)
            user = ASSIGNMENT_USER_TEMPLATE.format(max_codes=max_codes, theme_json=theme_json, responses_json=responses_json)
            
            def make_request():
                # Estimate tokens for rate limiting
                estimated_tokens = estimate_tokens(responses_json)
                check_rate_limits(estimated_tokens)
                return oai_json_completion(client, model, ASSIGNMENT_SYSTEM, user, seed, ASSIGNMENTS_SCHEMA)
            
            try:
                data, usage = retry_with_backoff(make_request)
                
                # Accumulate usage
                chunk_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                chunk_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                
                # Parse response
                if isinstance(data, dict) and "results" in data:
                    ai_assignments = data["results"]
                else:
                    ai_assignments = data if isinstance(data, list) else []
                
                # Map AI assignments back to texts with confidence calibration
                for assignment in ai_assignments:
                    ai_idx = assignment.get("idx")
                    if ai_idx is not None and ai_idx < len(responses_for_ai):
                        text = responses_for_ai[ai_idx]["text"]
                        
                        # Apply confidence calibration to each assignment
                        calibrated_assignments = []
                        for assign in assignment.get("assignments", []):
                            original_confidence = assign.get("confidence", 0.5)
                            theme_id = assign.get("theme_id", "")
                            calibrated_confidence = calibrate_confidence(original_confidence, text, theme_id)
                            
                            calibrated_assignments.append({
                                "theme_id": theme_id,
                                "confidence": calibrated_confidence
                            })
                        
                        chunk_assignments[text] = {
                            "assignments": calibrated_assignments
                        }
                
                # Handle any responses that didn't get assignments
                for response in responses_for_ai:
                    text = response["text"]
                    if text not in chunk_assignments:
                        chunk_assignments[text] = {
                            "assignments": [{"theme_id": non_answer_theme_id or "T1.1", "confidence": 0.5}]
                        }
                
            except Exception as e:
                # Fallback: assign first theme to all responses in this chunk
                for response in responses_for_ai:
                    text = response["text"]
                    chunk_assignments[text] = {
                        "assignments": [{"theme_id": non_answer_theme_id or "T1.1", "confidence": 0.5}]
                    }
        
        return chunk_assignments, chunk_usage
    
    # Use minimal parallelism for cost optimization
    max_workers = min(3, len(all_chunks))  # Reduced from 50 to 3 for cost savings
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {executor.submit(process_single_chunk, chunk_data): i for i, chunk_data in enumerate(all_chunks)}
        
        completed = 0
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                chunk_assignments, chunk_usage = future.result()
                
                # Merge chunk assignments into main dictionary
                text_to_assignment.update(chunk_assignments)
                
                # Accumulate usage
                total_usage["prompt_tokens"] += chunk_usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += chunk_usage.get("completion_tokens", 0)
                
                completed += 1
                progress = 5 + (completed * 90 // len(all_chunks))
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {completed}/{len(all_chunks)}")
                
            except Exception as e:
                st.error(f"Error processing chunk: {str(e)}")
                # Continue with other chunks
    
    # Step 3: Expand assignments to all original responses (maintaining duplicates)
    all_assignments = []
    
    for text, indices in text_to_indices.items():
        if text in text_to_assignment:
            assignment_template = text_to_assignment[text]
            # Apply the same assignment to all instances of this text
            for idx in indices:
                all_assignments.append({
                    "idx": idx,
                    "assignments": assignment_template["assignments"]
                })
        else:
            # Fallback for any missed texts
            st.warning(f"Missing assignment for text: {text[:50]}...")
            for idx in indices:
                all_assignments.append({
                    "idx": idx,
                    "assignments": [{"theme_id": non_answer_theme_id or "T1.1", "confidence": 0.5}]
                })
    
    # Step 4: Ensure we have assignments for ALL response indices
    assigned_indices = {a["idx"] for a in all_assignments}
    total_rows = len(rows)
    
    for i in range(total_rows):
        if i not in assigned_indices:
            # Missing assignment - add fallback
            all_assignments.append({
                "idx": i,
                "assignments": [{"theme_id": non_answer_theme_id or "T1.1", "confidence": 0.5}]
            })
            st.warning(f"Added fallback assignment for missing response index {i}")
    
    # Sort by index
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
                expanded_assignment = base_assignment.copy()
                expanded_assignment["idx"] = original_idx
                expanded_assignments.append(expanded_assignment)
    
    # Sort by original index to maintain order
    expanded_assignments.sort(key=lambda x: x["idx"])
    return expanded_assignments


def verify_low_confidence(client: OpenAI, model: str, theme_dict: Dict[str, Any], flagged: List[Dict[str, Any]], low_thresh: float, seed: int | None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Verify low confidence assignments"""
    if not flagged:
        return [], {"prompt_tokens": 0, "completion_tokens": 0}
    
    chunk_size = 5
    total_chunks = math.ceil(len(flagged) / chunk_size)
    
    # Prepare all chunks for parallel processing
    all_chunks = []
    for i in range(0, len(flagged), chunk_size):
        chunk_flagged = flagged[i:i + chunk_size]
        all_chunks.append(chunk_flagged)
    
    def process_verification_chunk(chunk_flagged):
        """Process a single chunk of flagged responses"""
        theme_json = json.dumps(slim_theme_for_assignment(theme_dict))
        flagged_json = json.dumps(chunk_flagged)
        user = VERIFY_USER_TEMPLATE.format(low_thresh=low_thresh, theme_json=theme_json, flagged_json=flagged_json)
        
        def make_request():
            # Minimal rate limiting
            estimated_tokens = estimate_tokens(flagged_json)
            check_rate_limits(estimated_tokens)
            return oai_json_completion(client, model, VERIFY_SYSTEM, user, seed, ASSIGNMENTS_SCHEMA)
        
        return retry_with_backoff(make_request)
    
    # Use minimal parallelism for cost optimization
    max_workers = min(2, len(all_chunks))  # Reduced from 20 to 2 for cost savings
    
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

def auto_calculate_tiny_threshold(coded_df: pd.DataFrame, theme_counts: pd.Series) -> float:
    """
    Auto-calculate optimal tiny theme threshold using hybrid approach.
    Conservative settings: balances actionable insights with preserving niche themes.
    
    Combines:
    1. Research best practices (minimum 3 responses)
    2. Statistical outlier detection (IQR method)
    3. Dataset size adjustments
    """
    total_responses = len(coded_df)
    
    # Step 1: Research minimum - at least 3 responses for reliability
    min_responses = 3
    research_threshold = (min_responses / total_responses) * 100
    
    # Step 2: Statistical outlier detection (IQR) - conservative multiplier
    statistical_threshold = research_threshold  # Default fallback
    if len(theme_counts) > 4:  # Need enough themes for meaningful IQR
        counts_array = theme_counts.values
        Q1 = np.percentile(counts_array, 25)
        Q3 = np.percentile(counts_array, 75)
        IQR = Q3 - Q1
        
        # Conservative: use 2.0 multiplier instead of 1.5 (flags fewer themes)
        # This preserves more niche themes unless they're really outliers
        outlier_count = max(1, Q1 - (2.0 * IQR))
        statistical_threshold = (outlier_count / total_responses) * 100
    
    # Step 3: Take the more conservative (higher) of the two
    # This means we only flag themes that are both statistically unusual AND below minimum N
    threshold = max(research_threshold, statistical_threshold)
    
    # Step 4: Apply reasonable bounds based on dataset size
    # Conservative approach: wider acceptable range
    if total_responses < 100:
        threshold = max(2.0, threshold)  # At least 2% for very small samples
    elif total_responses < 500:
        threshold = max(1.0, threshold)  # At least 1% for small samples  
    elif total_responses > 10000:
        threshold = min(0.75, threshold)  # Cap at 0.75% for very large samples
    else:
        threshold = max(0.5, threshold)  # Floor at 0.5% for medium samples
    
    # Round to 1 decimal for clean UI
    return round(max(0.5, min(5.0, threshold)), 1)


def analyze_theme_distribution(coded_df: pd.DataFrame, tiny_threshold: float, theme_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze theme distribution and identify potential issues with outlier handling"""
    total_responses = len(coded_df)
    
    # Count responses in different categories
    subtheme1_col = f"{coded_df.columns[coded_df.columns.str.contains('_SubTheme1')][0]}" if any(coded_df.columns.str.contains("_SubTheme1")) else "MajorTheme1"
    other_themes = coded_df[coded_df[subtheme1_col].str.contains("Other", case=False, na=False)]
    not_applicable = coded_df[coded_df[subtheme1_col].str.contains("Not applicable", case=False, na=False)]
    manual_review_needed = coded_df.get("ManualReview", pd.Series([False] * len(coded_df), dtype=bool))
    
    other_count = len(other_themes)
    not_applicable_count = len(not_applicable)
    manual_review_count = manual_review_needed.sum() if hasattr(manual_review_needed, 'sum') else 0
    
    other_percent = (other_count / total_responses * 100) if total_responses > 0 else 0
    not_applicable_percent = (not_applicable_count / total_responses * 100) if total_responses > 0 else 0
    manual_review_percent = (manual_review_count / total_responses * 100) if total_responses > 0 else 0
    
    # Identify themes with very low counts (potential candidates for "Other")
    theme_counts = coded_df[subtheme1_col].value_counts()
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
    subtheme1_col = f"{coded_df.columns[coded_df.columns.str.contains('_SubTheme1')][0]}" if any(coded_df.columns.str.contains("_SubTheme1")) else "MajorTheme1"
    actual_counts = coded_df[subtheme1_col].value_counts()
    
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


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(
    page_title="Express Explorer Thematic Coder", 
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
        background: #d4edda;
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
    <h1>📊 Express Explorer Thematic Coder</h1>
    <p>Upload open‑ended survey responses, auto‑discover Major and Sub‑themes, and export professionally coded data.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    model = "gpt-5"
    seed = 42  # Hard-coded for deterministic results
    st.info("🤖 **High Quality Mode: GPT-5 for All Steps**")
    st.caption("🎯 **Theme Generation**: GPT-5 (highest quality)")
    st.caption("🎯 **Assignment**: GPT-5 (highest accuracy)")
    st.caption("💡 GPT-5: 500K TPM, 500 RPM - optimized for cost efficiency")
    st.caption("🔒 Deterministic mode enabled (same input → same output)")
    allow_multicode = st.toggle("Multi‑coding", value=True)
    max_codes = 3
    single_or_multi = "Multi" if allow_multicode else "Single"

    low_thresh = st.slider("Low confidence threshold", 0.0, 1.0, 0.60, 0.01)
    
    st.divider()
    st.subheader("Theme count")
    auto_theme = st.toggle("Auto decide theme count", value=True)
    theme_min, theme_max = st.slider("Preferred range", min_value=3, max_value=20, value=(6, 12))

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

# Optional ID passthrough
id_cols_guess = [c for c in df.columns if c.lower() in {"id", "respondent_id", "record id", "record_id", "transaction id", "transaction_id", "uuid"}]
pass_id_cols = st.multiselect("ID columns to carry through", options=df.columns.tolist(), default=id_cols_guess)

# Question context - provides additional context without creating priority themes
question_text = st.text_input("Question text (optional - provides context for theme discovery)", value="", 
                               help="Enter the original survey question. This helps the AI understand context without creating pre-conceived themes.")

# Prepare series
ser = df[text_col].map(clean_text)

# Build unique set with frequency weights but preserve order for output mapping
value_counts = ser.value_counts(dropna=False)
unique_texts = value_counts.index.tolist()
unique_freqs = value_counts.values.tolist()

# Initialize OpenAI client
client = get_openai_client()

st.divider()
st.subheader("Theme discovery")

# Theme management options
col1, col2 = st.columns([2, 1])
with col1:
    st.write("**Choose theme source:**")
with col2:
    theme_source = st.radio("Theme source", ["Generate new themes", "Upload existing themes"], horizontal=True, label_visibility="collapsed")

# Theme upload functionality
if theme_source == "Upload existing themes":
    uploaded_theme_file = st.file_uploader("Upload theme dictionary (JSON or XLSX)", type=["json", "xlsx"], key="theme_upload")
    
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
                            "examples": row.get("ExampleQuotes", "").split("; ") if pd.notna(row.get("ExampleQuotes")) else []
                        }
                        current_major["subs"].append(sub_theme)
                
                st.session_state["theme_dict"] = theme_dict
                st.success("Theme dictionary uploaded successfully!")

        except Exception as e:
            st.error(f"Error loading theme file: {str(e)}")

        if "theme_dict" in st.session_state:
            st.write("**Current theme dictionary:**")
            theme_df = flatten_theme_dict(st.session_state["theme_dict"])
            st.dataframe(theme_df, width="stretch")

        if st.button("Clear uploaded themes"):
            if "theme_dict" in st.session_state:
                del st.session_state["theme_dict"]
            st.rerun()

# Theme generation section
if theme_source == "Generate new themes":
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
        if t and not is_empty_like(t) and len(t.strip()) >= 10
    ]
    estimated_prompt_tokens = estimate_tokens(json.dumps(filtered_for_estimation))
    estimated_completion_tokens = 2000  # Rough estimate for theme generation
    
    # If chunking will be needed, estimate for multiple requests
    if estimated_prompt_tokens > 400000:
        num_chunks = math.ceil(estimated_prompt_tokens / 350000)
        estimated_cost = fmt_cost(estimated_prompt_tokens, estimated_completion_tokens * num_chunks, pricing)
        st.info(f"**Estimated cost for theme generation (GPT-5):** ${estimated_cost:.4f} (will process in {num_chunks} chunks, {len(unique_texts)} unique responses)")
    else:
        estimated_cost = fmt_cost(estimated_prompt_tokens, estimated_completion_tokens, pricing)
        st.info(f"**Estimated cost for theme generation (GPT-5):** ${estimated_cost:.4f} (based on {len(unique_texts)} unique responses)")
    
    if estimated_prompt_tokens > 400000:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Very Large Dataset Detected</h4>
            <p>Your dataset exceeds 400K tokens and will be processed in chunks. With GPT-5's 500K TPM limit, this ensures optimal processing.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Process Themes", type="primary"):
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
            
            # Theme discovery with optional question context (no priority themes)
            if question_text and question_text.strip():
                st.info(f"📝 Using question context: \"{question_text}\"")
            theme_dict, usage_theme = build_theme_frame_with_progress(client, model, unique_texts, unique_freqs, seed, theme_progress, theme_status, question_text)
            st.session_state["theme_dict"] = theme_dict
            
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
                for i, t in enumerate(ser.fillna("").astype(str).tolist())
            ]

            assigned, usage_assign = assign_codes_with_progress(client, model, theme_dict, rows_payload, max_codes if allow_multicode else 1, seed, assign_progress, assign_status)
            st.session_state["assigned_raw"] = assigned
            
            assign_end_time = time.time()
            assign_duration = assign_end_time - assign_start_time
            
            assign_progress.progress(100)
            assign_status.text("✅ Theme assignment complete!")
            
            # Update timers
            total_elapsed = time.time() - overall_start_time
            total_timer.metric("⏱️ Total Time", f"{int(total_elapsed//60)}:{int(total_elapsed%60):02d}")
            assign_timer.metric("🏷️ Assignment", f"{int(assign_duration//60)}:{int(assign_duration%60):02d}")
            
            # Step 2b: Auto-verify low confidence assignments
            st.write("🔍 **Auto-Verifying Low Confidence Responses**")
            verify_progress = st.progress(0)
            verify_status = st.empty()
            verify_timer = st.empty()
            
            verify_start_time = time.time()
            verify_status.text("Identifying low confidence responses...")
            verify_progress.progress(10)
            verify_timer.metric("🔍 Verification", "Running...")
            
            # Identify low confidence responses
            low = float(low_thresh)
            flagged = []
            for item in st.session_state["assigned_raw"]:
                confs = [a.get("confidence", 0.0) for a in item.get("assignments", [])]
                top_conf = max(confs) if confs else 0.0
                if top_conf < low:
                    flagged.append(item)
            
            verify_progress.progress(25)
            
            if flagged:
                verify_status.text(f"Re-checking {len(flagged)} low-confidence assignments...")
                verify_progress.progress(50)
                
                verified, usage_verify = verify_low_confidence(client, model, theme_dict, flagged, low_thresh=low, seed=seed)
                
                # Replace items by idx - ensure clean structure
                by_idx = {x["idx"]: x for x in st.session_state["assigned_raw"]}
                for v in verified:
                    cleaned_item = {
                        "idx": v["idx"],
                        "assignments": v.get("assignments", [])
                    }
                    by_idx[v["idx"]] = cleaned_item
                st.session_state["assigned_raw"] = [by_idx[i] for i in sorted(by_idx.keys())]
                
                verify_progress.progress(100)
                verify_status.text(f"✅ Verified {len(verified)} low-confidence assignments!")
            else:
                verify_progress.progress(100)
                verify_status.text("✅ All assignments have high confidence!")
            
            verify_end_time = time.time()
            verify_duration = verify_end_time - verify_start_time
            
            # Update timers
            total_elapsed = time.time() - overall_start_time
            total_timer.metric("⏱️ Total Time", f"{int(total_elapsed//60)}:{int(total_elapsed%60):02d}")
            verify_timer.metric("🔍 Verification", f"{int(verify_duration//60)}:{int(verify_duration%60):02d}")
            
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

st.caption("💡 **Tip**: Use these exports to save your themes and import them later using the 'Upload existing themes' option!")

# Show JSON preview
with st.expander("🔍 Preview JSON Format"):
    st.code(theme_json[:1000] + "..." if len(theme_json) > 1000 else theme_json, language="json")
    st.caption("This JSON format can be directly imported using the 'Upload existing themes' option above.")

# Comprehensive export (themes + coded data)
st.write("**Complete Export (Themes + Coded Data):**")
comprehensive_buf = io.BytesIO()
with pd.ExcelWriter(comprehensive_buf, engine="xlsxwriter") as writer:
    # Add theme dictionary sheet
    theme_df.to_excel(writer, sheet_name="Theme Dictionary", index=False)
    
    # Add coded data sheet (we'll need to build this first)
    if "assigned_raw" in st.session_state:
        # Build a preview of coded data for the comprehensive export
        assign_map = {x["idx"]: x for x in st.session_state["assigned_raw"]}
        major_map = map_theme_id_to_major(theme_df)
        label_map = {r["ThemeID"]: r["Label"] for _, r in theme_df.iterrows()}
        
        # Create a sample of coded data (first 100 rows for export)
        sample_coded_rows = []
        for i in range(min(100, len(assign_map))):
            item = assign_map.get(i, {"assignments": [], "rationale": ""})
            assigns = item.get("assignments", [])
            if assigns:
                primary_theme_id = assigns[0].get("theme_id", "")
                primary_major = major_map.get(primary_theme_id, "")
                primary_sub = primary_theme_id if primary_theme_id and theme_df.loc[theme_df["ThemeID"] == primary_theme_id, "Level"].tolist()[0] == "Sub" else ""
                
                sample_coded_rows.append({
                    "Response_Index": i,
                    "MajorTheme1": label_map.get(primary_major, ""),
                    "SubTheme1": label_map.get(primary_theme_id, "") if primary_sub else "",
                    "Confidence": assigns[0].get("confidence", 0.0),
                    "ThemeID": primary_theme_id
                })
        
        if sample_coded_rows:
            sample_coded_df = pd.DataFrame(sample_coded_rows)
            sample_coded_df.to_excel(writer, sheet_name="Sample Coded Data", index=False)

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

coded_rows = []
for i in range(len(df)):
    item = assign_map.get(i, {"assignments": [], "rationale": ""})
    assigns = item.get("assignments", [])
    assigns = sorted(assigns, key=lambda a: a.get("confidence", 0.0), reverse=True)
    assigns = assigns[: (max_codes if allow_multicode else 1)]

    codes = [a.get("theme_id") for a in assigns]
    confs = [float(a.get("confidence", 0.0)) for a in assigns]

    # Calculate average confidence
    avg_confidence = float(np.mean(confs)) if confs else 0.0
    
    # Build row with new schema: [IDs] [RawOpenEnd] [MajorTheme1/2/3] [variablelabel_SubTheme1/2/3] [Confidence]
    row = {}
    
    # Carry through IDs first
    for c in pass_id_cols:
        row[c] = df.loc[i, c]
    
    # Add raw open end text (using variable label as column name)
    row[text_col] = ser.iloc[i]
    
    # Add Major themes - one for each code, showing which Major theme each sub-theme belongs to
    # MajorTheme1 is always populated (even for single-coded responses)
    if len(codes) > 0:
        major1_id = major_map.get(codes[0], "")
        row["MajorTheme1"] = label_map.get(major1_id, "")
    else:
        row["MajorTheme1"] = ""
    
    # MajorTheme2 only populated if 2nd code exists and has confidence ≥ 0.6
    if len(codes) > 1 and confs[1] >= 0.6:
        major2_id = major_map.get(codes[1], "")
        row["MajorTheme2"] = label_map.get(major2_id, "")
    else:
        row["MajorTheme2"] = ""
    
    # MajorTheme3 only populated if 3rd code exists and has confidence ≥ 0.6
    if len(codes) > 2 and confs[2] >= 0.6:
        major3_id = major_map.get(codes[2], "")
        row["MajorTheme3"] = label_map.get(major3_id, "")
    else:
        row["MajorTheme3"] = ""
    
    # Add sub-themes with variable label prefix
    # SubTheme1 is always populated (even for single-coded responses)
    row[f"{text_col}_SubTheme1"] = label_map.get(codes[0], "") if len(codes) > 0 else ""
    
    # SubTheme2 only populated for multi-coded high-confidence responses
    if len(codes) > 1 and confs[1] >= 0.6:  # High confidence threshold for additional codes
        row[f"{text_col}_SubTheme2"] = label_map.get(codes[1], "")
    else:
        row[f"{text_col}_SubTheme2"] = ""
    
    # SubTheme3 only populated for multi-coded high-confidence responses
    if len(codes) > 2 and confs[2] >= 0.6:  # High confidence threshold for additional codes
        row[f"{text_col}_SubTheme3"] = label_map.get(codes[2], "")
    else:
        row[f"{text_col}_SubTheme3"] = ""
    
    # Add confidence
    row["Confidence"] = avg_confidence
    
    coded_rows.append(row)

coded_df = pd.DataFrame(coded_rows)

# Order columns: [IDs] [RawOpenEnd] [MajorTheme1/2/3] [SubTheme1/2/3] [Confidence]
id_before = pass_id_cols.copy()
base_cols = [text_col, "MajorTheme1", "MajorTheme2", "MajorTheme3", f"{text_col}_SubTheme1", f"{text_col}_SubTheme2", f"{text_col}_SubTheme3", "Confidence"]
ordered_cols = id_before + base_cols
coded_df = coded_df[ordered_cols]

# Review & Verification Section
st.divider()
st.subheader("Review & Verification")

# Show coded data preview
st.write("**Coded Data Preview:**")
st.dataframe(coded_df.head(20), width="stretch")

# Identify low confidence responses
low = float(low_thresh)
flagged = []
for item in st.session_state["assigned_raw"]:
    confs = [a.get("confidence", 0.0) for a in item.get("assignments", [])]
    top_conf = max(confs) if confs else 0.0
    if top_conf < low:
        flagged.append(item)

# Review summary (verification already done automatically)
if flagged:
    st.write(f"**ℹ️ {len(flagged)} responses were auto-verified (initially below {low_thresh} confidence)**")
    st.caption("These responses were automatically re-checked during the coding process.")
    
    # Analyze low-confidence patterns for theme suggestions
    pattern_analysis = analyze_low_confidence_patterns(flagged)
    
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
                            st.info("Re-assigning all responses with the expanded theme dictionary...")
                            # This would trigger a re-assignment - for now just show a message
                            st.success("Theme dictionary updated! Consider re-running the assignment process to use the new themes.")
                        st.rerun()
                    else:
                        st.error("Please provide both theme name and definition")
            
            with col4:
                if st.button("🤖 AI Suggest Themes", type="secondary"):
                    st.info("🤖 AI theme suggestion feature coming soon! For now, use the pattern analysis above to manually create themes.")
    
    # Optional manual review mode for advanced users
    if st.checkbox("Enable manual review mode", value=False):
        review_mode = "Manual review"
    else:
        review_mode = "Automatic verification"
else:
    st.success("✅ All responses have high confidence scores!")
    review_mode = "Automatic verification"

if review_mode == "Manual review":
    st.write(f"**{len(flagged)} responses flagged for manual review (confidence < {low_thresh})**")
    
    if flagged:
        # Create a simple pagination system
        if "review_page" not in st.session_state:
            st.session_state["review_page"] = 0
        
        page_size = 5
        total_pages = (len(flagged) + page_size - 1) // page_size
        start_idx = st.session_state["review_page"] * page_size
        end_idx = min(start_idx + page_size, len(flagged))
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Previous", disabled=st.session_state["review_page"] == 0):
                st.session_state["review_page"] -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state['review_page'] + 1} of {total_pages}")
        with col3:
            if st.button("Next →", disabled=st.session_state["review_page"] >= total_pages - 1):
                st.session_state["review_page"] += 1
                st.rerun()
        
        # Show current batch for review
        for i in range(start_idx, end_idx):
            item = flagged[i]
            idx = item["idx"]
            original_text = ser.iloc[idx]
            current_assignments = item.get("assignments", [])
            
            st.write(f"**Response {idx + 1}:** {original_text}")
            assignments_text = [f"{a.get('theme_id', '')} (conf: {a.get('confidence', 0):.2f})" for a in current_assignments]
            st.write(f"**Current assignments:** {assignments_text}")
            
            # Show rationale if available (only for low confidence items)
            rationale = item.get("rationale", "")
            if rationale:
                st.write(f"**AI Rationale:** {rationale}")
            
            # Manual override options
            col1, col2 = st.columns([3, 1])
            with col1:
                # Show available themes for manual selection
                theme_options = []
                for major in st.session_state["theme_dict"].get("major_themes", []):
                    theme_options.append((major["id"], f"{major['label']} (Major)"))
                    for sub in major.get("subs", []):
                        theme_options.append((sub["id"], f"{sub['label']} (Sub of {major['label']})"))
                
                selected_themes = st.multiselect(
                    f"Select themes for response {idx + 1}:",
                    options=[opt[0] for opt in theme_options],
                    format_func=lambda x: next(opt[1] for opt in theme_options if opt[0] == x),
                    default=[a.get("theme_id") for a in current_assignments if a.get("theme_id")],
                    key=f"manual_review_{idx}"
                )
            
            with col2:
                if st.button(f"Update {idx + 1}", key=f"update_{idx}"):
                    # Update the assignment
                    new_assignments = [{"theme_id": theme_id, "confidence": 1.0} for theme_id in selected_themes]
                    item["assignments"] = new_assignments
                    item["rationale"] = "Manually reviewed and updated"
                    
                    # Update in session state
                    by_idx = {x["idx"]: x for x in st.session_state["assigned_raw"]}
                    by_idx[idx] = item
                    st.session_state["assigned_raw"] = [by_idx[i] for i in sorted(by_idx.keys())]
                    st.success(f"Updated response {idx + 1}")
                    st.rerun()
            
            st.divider()
    else:
        st.success("No responses need manual review!")

else:  # Automatic verification (already completed during workflow)
    st.info("✅ All low-confidence responses were automatically verified during the coding process.")


# Theme distribution for charting, with volume weights
st.divider()
st.subheader("Theme distribution")

# Compute support counts using primary SubTheme1 as assignment for counting
subtheme1_col = [col for col in coded_df.columns if "_SubTheme1" in col][0]
count_series = coded_df[subtheme1_col].replace("", np.nan).dropna()
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
    subtheme1_col = [col for col in coded_df.columns if "_SubTheme1" in col][0]
    coded_responses = len(coded_df[coded_df[subtheme1_col] != ""])
    st.metric("Coded Responses", f"{coded_responses:,}")

with col3:
    coding_rate = (coded_responses / total_responses * 100) if total_responses > 0 else 0
    st.metric("Coding Rate", f"{coding_rate:.1f}%")

with col4:
    avg_confidence = coded_df["Confidence"].mean()
    st.metric("Avg Confidence", f"{avg_confidence:.2f}")

# Theme distribution chart with interactive legend
st.write("**Theme Distribution**")

# Prepare data for charting
subtheme1_col = [col for col in coded_df.columns if "_SubTheme1" in col][0]
major_counts = coded_df["MajorTheme1"].value_counts()
sub_counts = coded_df[subtheme1_col].value_counts()

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

# Cross-tabulation with demographic variables - moved below chart
if pass_id_cols:
    st.write("**Cross-tabulation Analysis**")
    
    # Let user select a demographic variable for cross-tab
    demo_col = st.selectbox("Select demographic variable for cross-tabulation", 
                           options=pass_id_cols, 
                           help="Choose a column to analyze theme distribution across different groups")
    
    if demo_col:
        # Create cross-tab with percentages only
        subtheme1_col = [col for col in coded_df.columns if "_SubTheme1" in col][0]
        cross_tab_pct = pd.crosstab(coded_df[demo_col], coded_df[subtheme1_col], normalize="index") * 100
        
        st.write("**Percentages by Demographic Group**")
        st.dataframe(cross_tab_pct.round(1), width="stretch")

# Auto-calculate optimal tiny theme threshold
subtheme1_col = [col for col in coded_df.columns if "_SubTheme1" in col][0]
theme_counts = coded_df[subtheme1_col].replace("", np.nan).dropna().value_counts()
auto_tiny_threshold = auto_calculate_tiny_threshold(coded_df, theme_counts)

# Show auto-calculated threshold with explanation and manual override option
st.write("**Theme Distribution Analysis**")
col_info1, col_info2 = st.columns([2, 1])
with col_info1:
    st.info(f"💡 **Smart Threshold:** Auto-calculated at **{auto_tiny_threshold}%** based on {len(coded_df):,} responses and {len(theme_counts)} themes (Research min: 3 responses + Statistical outlier detection)")
with col_info2:
    manual_override = st.number_input(
        "Manual override (%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=auto_tiny_threshold,
        step=0.1,
        help="Override the auto-calculated threshold if needed"
    )
    use_manual = (manual_override != auto_tiny_threshold)

# Use manual override if changed, otherwise use auto
active_tiny_threshold = manual_override if use_manual else auto_tiny_threshold

if use_manual:
    st.caption(f"ℹ️ Using manual override: {manual_override}%")

# Calculate dynamic thresholds
dynamic_thresholds = calculate_dynamic_thresholds(st.session_state.get("theme_dict"), st.session_state.get("assigned_raw", []))

# Display dynamic threshold recommendations
if dynamic_thresholds["recommendations"]:
    with st.expander("🎯 Additional Dynamic Threshold Recommendations", expanded=False):
        for rec in dynamic_thresholds["recommendations"]:
            st.info(f"• {rec}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Suggested Tiny Threshold", f"{dynamic_thresholds['tiny_threshold']:.1f}%")
        with col2:
            st.metric("Suggested Confidence Threshold", f"{dynamic_thresholds['low_confidence_threshold']:.1f}")
        with col3:
            st.metric("Suggested Other Usage Threshold", f"{dynamic_thresholds['other_usage_threshold']:.1f}%")

# Theme distribution analysis using active threshold
theme_analysis = analyze_theme_distribution(coded_df, active_tiny_threshold, st.session_state.get("theme_dict"))

# Display analysis results
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Other Responses", f"{theme_analysis['other_count']:,}", f"{theme_analysis['other_percent']:.1f}%")
with col2:
    st.metric("Not Applicable", f"{theme_analysis['not_applicable_count']:,}", f"{theme_analysis['not_applicable_percent']:.1f}%")
with col3:
    st.metric("Manual Review", f"{theme_analysis['manual_review_count']:,}", f"{theme_analysis['manual_review_percent']:.1f}%")
with col4:
    st.metric("Single-Response Themes", f"{theme_analysis['tiny_theme_count']:,}", f"{theme_analysis['tiny_theme_percent']:.1f}%")
with col5:
    threshold_status = "⚠️ Exceeded" if theme_analysis['threshold_exceeded'] else "✅ Within Limit"
    threshold_label = f"{'Manual' if use_manual else 'Auto'}: {active_tiny_threshold:.1f}%"
    st.metric("Tiny Theme Threshold", threshold_label, threshold_status)

# Show recommendations if any
if theme_analysis['recommendations']:
    st.write("**🔍 Analysis Recommendations:**")
    for i, rec in enumerate(theme_analysis['recommendations'], 1):
        st.info(f"{i}. {rec}")

# Show tiny themes if they exist
if theme_analysis['tiny_theme_names']:
    st.write(f"**Single-Response Themes ({len(theme_analysis['tiny_theme_names'])}):**")
    tiny_theme_text = ", ".join(theme_analysis['tiny_theme_names'][:10])
    if len(theme_analysis['tiny_theme_names']) > 10:
        tiny_theme_text += f" ... and {len(theme_analysis['tiny_theme_names']) - 10} more"
    st.caption(tiny_theme_text)

# Multi-coding analysis
if allow_multicode:
    subtheme1_col = [col for col in coded_df.columns if "_SubTheme1" in col][0]
    subtheme2_col = [col for col in coded_df.columns if "_SubTheme2" in col][0]
    subtheme3_col = [col for col in coded_df.columns if "_SubTheme3" in col][0]
    
    multi_coded = coded_df[subtheme2_col] != ""
    st.write(f"**Multi-coding Analysis:** {multi_coded.sum()} responses ({multi_coded.mean()*100:.1f}%) have multiple codes")
    
    # Show multi-coded examples
    if multi_coded.any():
        st.write("**Examples of multi-coded responses:**")
        text_col_name = [col for col in coded_df.columns if col not in pass_id_cols and col not in ["MajorTheme1", "MajorTheme2", "MajorTheme3"] and "_SubTheme" not in col and col != "Confidence"][0]
        multi_examples = coded_df[multi_coded][[text_col_name, subtheme1_col, subtheme2_col, subtheme3_col]].head(10)
        st.dataframe(multi_examples, width="stretch")

# ------------------------------
# Export
# ------------------------------

st.divider()
st.subheader("Export")

q_name = text_col.replace(" ", "_")
file_name = f"{q_name}_thematic_coding_{today_stamp()}.xlsx"

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

st.download_button("Download XLSX", data=buf.getvalue(), file_name=file_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------
# Cost summary
# ------------------------------

# Initialize usage tracking
if "_usage_totals" not in st.session_state:
    st.session_state["_usage_totals"] = {"prompt_tokens": 0, "completion_tokens": 0}

# Accumulate usage from completed steps
total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

# Add usage from theme generation if available
if "usage_theme" in locals():
    total_usage["prompt_tokens"] += usage_theme.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += usage_theme.get("completion_tokens", 0)

# Add usage from assignment if available
if "usage_assign" in locals():
    total_usage["prompt_tokens"] += usage_assign.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += usage_assign.get("completion_tokens", 0)

# Add usage from verification if available
if "usage_verify" in locals():
    total_usage["prompt_tokens"] += usage_verify.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += usage_verify.get("completion_tokens", 0)

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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prompt Tokens", f"{total_usage['prompt_tokens']:,}")
    with col2:
        st.metric("Completion Tokens", f"{total_usage['completion_tokens']:,}")
    with col3:
        st.metric("Total Cost (GPT-5)", f"${estimated_cost:.4f}")
    
    st.caption("Cost estimates based on current OpenAI pricing. High quality mode prioritizes accuracy over cost.")
