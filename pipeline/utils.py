from __future__ import annotations

import json
from typing import Any, List


def estimate_tokens(text: str, model: str = "gpt-5") -> int:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def safe_prompt_token_budget(model: str, reserve_output_tokens: int = 8_000) -> int:
    context_window = 128_000
    max_output_tokens = 16_384
    if model.startswith("gpt-5"):
        context_window = 400_000
        max_output_tokens = 128_000
    max_input = max(1, context_window - max_output_tokens)
    return max(1, int(max_input * 0.85) - reserve_output_tokens)


def chunk_data(data: list, max_tokens: int, model: str = "gpt-5") -> list:
    chunks = []
    current_chunk = []
    current_tokens = 0

    for item in data:
        item_str = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        item_tokens = estimate_tokens(item_str, model=model)

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
