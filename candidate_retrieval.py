from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from theme_governance import normalize_theme_dict_order


def build_subtheme_records(theme_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for major in theme_dict.get("major_themes", []) or []:
        major_label = major.get("label", "")
        for sub in major.get("subs", []) or []:
            label = sub.get("label", "")
            definition = sub.get("definition", "")
            examples = sub.get("examples", []) or []
            records.append({
                "id": sub.get("id", ""),
                "label": label,
                "definition": definition,
                "example": examples[0] if examples else "",
                "text": f"{major_label}::{label} — {definition}".strip(),
            })
    return records


def theme_signature(theme_dict: Dict[str, Any]) -> str:
    normalized = normalize_theme_dict_order(theme_dict)
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_texts(client, model: str, texts: List[str], batch_size: int = 128) -> Tuple[List[List[float]], int]:
    embeddings: List[List[float]] = []
    total_input_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        data = getattr(resp, "data", [])
        data_sorted = sorted(data, key=lambda d: d.index)
        embeddings.extend([d.embedding for d in data_sorted])
        usage = getattr(resp, "usage", None)
        if usage is not None:
            total_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
    return embeddings, total_input_tokens


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def get_candidate_ids(
    response_embeddings: np.ndarray,
    subtheme_embeddings: np.ndarray,
    subtheme_ids: List[str],
    top_k: int,
) -> List[List[str]]:
    if response_embeddings.size == 0 or subtheme_embeddings.size == 0:
        return [[] for _ in range(response_embeddings.shape[0])]

    subtheme_embeddings = normalize_vectors(subtheme_embeddings)
    response_embeddings = normalize_vectors(response_embeddings)

    scores = response_embeddings @ subtheme_embeddings.T
    candidates: List[List[str]] = []
    for row in scores:
        top_indices = np.argsort(-row)[:top_k]
        candidates.append([subtheme_ids[i] for i in top_indices])
    return candidates
