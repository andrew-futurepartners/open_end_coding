import os
import tempfile

from pipeline.llm_client import (
    CacheStats,
    RateLimiter,
    SQLiteCache,
    build_cache_key,
    compute_prompt_version,
    oai_json_completion,
)
from pipeline.run_audit import compute_run_id, dataset_signature


def test_cache_key_stability():
    key1 = build_cache_key("gpt-5", "sys", "user", None, 42, "v1")
    key2 = build_cache_key("gpt-5", "sys", "user", None, 42, "v1")
    key3 = build_cache_key("gpt-5", "sys2", "user", None, 42, "v1")
    assert key1 == key2
    assert key1 != key3


def test_cache_hit_no_api_call():
    tmpdir = tempfile.mkdtemp()
    cache_path = os.path.join(tmpdir, "cache.sqlite")
    cache = SQLiteCache(cache_path)
    cache_stats = CacheStats()
    limiter = RateLimiter(1000, 100000)

    system = "sys"
    user = "user"
    prompt_version = compute_prompt_version(system, user)
    key = build_cache_key("gpt-5", system, user, None, 42, prompt_version)
    cache.set(key, "gpt-5", prompt_version, "{\"ok\":true}", {"ok": True}, {"prompt_tokens": 1, "completion_tokens": 1})

    class FakeClient:
        pass

    data, usage, raw, from_cache = oai_json_completion(
        FakeClient(),
        "gpt-5",
        system,
        user,
        42,
        response_schema=None,
        limiter=limiter,
        cache=cache,
        cache_stats=cache_stats,
    )
    assert from_cache is True
    assert data == {"ok": True}


def test_rate_limiter_enforces_window():
    current = {"t": 0.0}

    def time_fn():
        return current["t"]

    def sleep_fn(seconds: float):
        current["t"] += seconds

    limiter = RateLimiter(rpm=1, tpm=1000, time_fn=time_fn, sleep_fn=sleep_fn)
    limiter.acquire(1)
    t1 = current["t"]
    limiter.acquire(1)
    t2 = current["t"]
    assert t2 - t1 >= 60.0


def test_run_id_deterministic():
    texts = ["a", "b", "a"]
    sig = dataset_signature(texts, "Q1")
    run_id1 = compute_run_id(sig, "gpt-5", 42, {"p": "v"}, {"k": 1})
    run_id2 = compute_run_id(sig, "gpt-5", 42, {"p": "v"}, {"k": 1})
    assert run_id1 == run_id2


if __name__ == "__main__":
    test_cache_key_stability()
    test_cache_hit_no_api_call()
    test_rate_limiter_enforces_window()
    test_run_id_deterministic()
    print({"status": "ok"})
