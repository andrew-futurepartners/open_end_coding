from __future__ import annotations

import atexit
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from openai import APITimeoutError
from openai import OpenAI
from openai import RateLimitError

from theme_governance import validate_json_schema


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    saved_prompt_tokens: int = 0
    saved_completion_tokens: int = 0


class RateLimiter:
    def __init__(self, rpm: int, tpm: int, time_fn=time.time, sleep_fn=time.sleep):
        self.rpm = max(1, int(rpm))
        self.tpm = max(1, int(tpm))
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._lock = threading.Lock()
        self._req_times = []
        self._tok_times = []
        self._tok_sum = 0

    def acquire(self, estimated_tokens: int) -> None:
        while True:
            wait = 0.0
            with self._lock:
                now = self._time_fn()
                cutoff = now - 60.0

                self._req_times = [t for t in self._req_times if t >= cutoff]
                self._tok_times = [(ts, toks) for ts, toks in self._tok_times if ts >= cutoff]
                self._tok_sum = sum(toks for _, toks in self._tok_times)

                if len(self._req_times) >= self.rpm:
                    wait = max(wait, (self._req_times[0] + 60.0) - now)

                if estimated_tokens > 0 and (self._tok_sum + estimated_tokens) > self.tpm:
                    need_to_expire = (self._tok_sum + estimated_tokens) - self.tpm
                    running = 0
                    for ts, toks in self._tok_times:
                        running += toks
                        if running >= need_to_expire:
                            wait = max(wait, (ts + 60.0) - now)
                            break

                if wait <= 0:
                    self._req_times.append(now)
                    if estimated_tokens > 0:
                        self._tok_times.append((now, estimated_tokens))
                        self._tok_sum += estimated_tokens
                    return
            self._sleep_fn(max(0.01, wait))


def _schema_to_responses_text_format(response_schema: Dict[str, Any]) -> Dict[str, Any]:
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
    return response_schema


def _should_use_responses_api(client: OpenAI, model: str) -> bool:
    return hasattr(client, "responses") and model.startswith("gpt-5")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schema_hash(response_schema: Dict[str, Any] | None) -> str:
    if response_schema is None:
        return "none"
    payload = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
    return _hash_text(payload)


def compute_prompt_version(system: str, user: str) -> str:
    return _hash_text(system + "\n" + user)


class SQLiteCache:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                prompt_version TEXT,
                created_at REAL,
                raw_text TEXT,
                parsed_json TEXT,
                usage_json TEXT,
                retry_json TEXT
            )
            """
        )
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            self._local.conn = conn
        return conn

    def get(self, cache_key: str) -> Dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "SELECT raw_text, parsed_json, usage_json FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            raw_text, parsed_json, usage_json = row
            return {
                "raw_text": raw_text,
                "parsed_json": json.loads(parsed_json) if parsed_json else None,
                "usage": json.loads(usage_json) if usage_json else {},
            }

    def set(self, cache_key: str, model: str, prompt_version: str, raw_text: str, parsed: Dict[str, Any],
            usage: Dict[str, Any], retry_meta: Dict[str, Any] | None = None) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, model, prompt_version, created_at, raw_text, parsed_json, usage_json, retry_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    model,
                    prompt_version,
                    time.time(),
                    raw_text,
                    json.dumps(parsed, ensure_ascii=False),
                    json.dumps(usage or {}),
                    json.dumps(retry_meta or {}),
                ),
            )
            conn.commit()


class JsonParseError(ValueError):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


def build_cache_key(
    model: str,
    system: str,
    user: str,
    response_schema: Dict[str, Any] | None,
    seed: int,
    prompt_version: str,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "user": user,
            "schema_hash": _schema_hash(response_schema),
            "seed": seed,
            "prompt_version": prompt_version,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _hash_text(payload)


def oai_json_completion(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    seed: int,
    response_schema: Dict[str, Any] | None,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    reasoning_effort: str | None = "minimal",
    verbosity: str | None = "low",
    reserve_output_tokens: int = 8_000,
) -> Tuple[Dict[str, Any], Dict[str, int], str, bool]:
    if model.startswith("gpt-5.2") and reasoning_effort == "minimal":
        reasoning_effort = "low"

    prompt_version = compute_prompt_version(system, user)
    cache_key = build_cache_key(model, system, user, response_schema, seed, prompt_version)

    cached = cache.get(cache_key)
    if cached is not None:
        cache_stats.hits += 1
        usage = cached.get("usage") or {}
        cache_stats.saved_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        cache_stats.saved_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        return cached.get("parsed_json") or {}, usage, cached.get("raw_text") or "", True

    cache_stats.misses += 1

    schema_str = ""
    if response_schema:
        schema_str = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
    estimated_tokens = max(1, (len(system) + len(user) + len(schema_str)) // 4)

    limiter.acquire(estimated_tokens)

    if _should_use_responses_api(client, model):
        text_format = _schema_to_responses_text_format(response_schema) if response_schema else {"type": "json_object"}
        response_params = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "reasoning": {"effort": reasoning_effort} if reasoning_effort else None,
            "text": {"format": text_format, "verbosity": verbosity} if verbosity else {"format": text_format},
            "seed": seed,
            "max_output_tokens": reserve_output_tokens,
        }

        try:
            resp = client.responses.create(**response_params)
        except TypeError as e:
            if "seed" in response_params:
                response_params.pop("seed", None)
                resp = client.responses.create(**response_params)
            else:
                raise e

        raw = getattr(resp, "output_text", None)
        if not raw:
            raw = json.dumps(resp.model_dump(), ensure_ascii=False)
        try:
            parsed = json.loads(raw)
        except Exception as e:
            raise JsonParseError("Failed to parse JSON output.", raw_text=raw) from e
        usage = getattr(resp, "usage", None)
        usage_dict = {
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
    else:
        request_params: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "seed": seed,
        }

        if response_schema is not None:
            request_params["response_format"] = response_schema
        else:
            request_params["response_format"] = {"type": "json_object"}

        request_params["max_completion_tokens"] = reserve_output_tokens
        response = client.chat.completions.create(**request_params)
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
        except Exception as e:
            raise JsonParseError("Failed to parse JSON output.", raw_text=content) from e
        raw = content
        usage_dict = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    cache.set(cache_key, model, prompt_version, raw, parsed, usage_dict, retry_meta={})
    return parsed, usage_dict, raw, False


def retry_with_backoff(func, max_retries: int = 6, base_delay: float = 0.5, max_delay: float = 30.0,
                       on_retry: Callable[[str], None] | None = None):
    transient_status = {408, 409, 425, 429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            status = getattr(e, "status_code", None)
            is_transient = isinstance(e, (RateLimitError, APITimeoutError)) or (status in transient_status)
            msg = str(e).lower()
            if ("timeout" in msg) or ("temporarily" in msg) or ("connection" in msg):
                is_transient = True
            if (not is_transient) or (attempt >= max_retries):
                raise
            if on_retry:
                on_retry(msg)
            delay = min(max_delay, base_delay * (2 ** attempt))
            time.sleep(delay)


def repair_json_to_schema(
    client: OpenAI,
    model: str,
    invalid_json: str,
    schema: Dict[str, Any],
    seed: int,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    context_label: str,
) -> Tuple[Dict[str, Any], Dict[str, int], str]:
    schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    system = "You are a JSON repair tool that returns strictly valid JSON."
    user = (
        "You will fix invalid JSON so it conforms exactly to the provided JSON schema.\n"
        "Return corrected JSON only. No commentary, no code fences.\n\n"
        f"Invalid JSON:\n{invalid_json}\n\n"
        f"JSON schema:\n{schema_json}\n"
    )
    data, usage, raw, _ = oai_json_completion(
        client,
        model,
        system,
        user,
        seed,
        response_schema=schema,
        limiter=limiter,
        cache=cache,
        cache_stats=cache_stats,
        reasoning_effort="minimal",
        verbosity="low",
        reserve_output_tokens=4_000,
    )
    return data, usage, raw


def validate_or_repair_json(
    client: OpenAI,
    model: str,
    parsed: Dict[str, Any],
    raw_text: str,
    schema: Dict[str, Any],
    seed: int,
    limiter: RateLimiter,
    cache: SQLiteCache,
    cache_stats: CacheStats,
    context_label: str,
) -> Tuple[Dict[str, Any], Dict[str, int], bool]:
    ok, err = validate_json_schema(parsed, schema)
    if ok:
        return parsed, {"prompt_tokens": 0, "completion_tokens": 0}, False

    data, usage, _ = repair_json_to_schema(
        client,
        model,
        raw_text or json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
        schema,
        seed,
        limiter,
        cache,
        cache_stats,
        context_label,
    )
    ok2, err2 = validate_json_schema(data, schema)
    if not ok2:
        raise ValueError(f"{context_label}: repair failed schema validation: {err2}")
    return data, usage, True
