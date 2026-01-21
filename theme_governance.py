import copy
import json
import re
from typing import Any, Dict, List, Tuple


THEME_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "theme_dictionary",
        "schema": {
            "type": "object",
            "properties": {
                "major_themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "definition": {"type": "string"},
                            "approx_pct": {"type": "number", "minimum": 0, "maximum": 1},
                            "subs": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "label": {"type": "string"},
                                        "definition": {"type": "string"},
                                        "approx_pct": {"type": "number", "minimum": 0, "maximum": 1},
                                        "examples": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                                    },
                                    "required": ["id", "label", "definition", "approx_pct", "examples"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["id", "label", "definition", "approx_pct", "subs"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["major_themes"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def _extract_schema(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not schema_dict:
        return {}
    if schema_dict.get("type") == "json_schema" and "json_schema" in schema_dict:
        return schema_dict["json_schema"].get("schema", {})
    return schema_dict


def validate_json_schema(obj: Any, schema_dict: Dict[str, Any]) -> Tuple[bool, str | None]:
    schema = _extract_schema(schema_dict)

    def _fail(path: str, msg: str) -> Tuple[bool, str]:
        return False, f"{path}: {msg}"

    def _validate(value: Any, schema_node: Dict[str, Any], path: str) -> Tuple[bool, str | None]:
        if not schema_node:
            return True, None

        if "enum" in schema_node:
            if value not in schema_node["enum"]:
                return _fail(path, f"expected one of {schema_node['enum']}, got {value!r}")

        node_type = schema_node.get("type")
        if node_type == "object":
            if not isinstance(value, dict):
                return _fail(path, "expected object")
            props = schema_node.get("properties", {})
            required = schema_node.get("required", [])
            for req in required:
                if req not in value:
                    return _fail(path, f"missing required property '{req}'")
            for key, val in value.items():
                if key in props:
                    ok, err = _validate(val, props[key], f"{path}.{key}")
                    if not ok:
                        return ok, err
                else:
                    if schema_node.get("additionalProperties") is False:
                        return _fail(path, f"unexpected property '{key}'")
            return True, None
        if node_type == "array":
            if not isinstance(value, list):
                return _fail(path, "expected array")
            min_items = schema_node.get("minItems")
            if isinstance(min_items, int) and len(value) < min_items:
                return _fail(path, f"expected at least {min_items} items")
            max_items = schema_node.get("maxItems")
            if isinstance(max_items, int) and len(value) > max_items:
                return _fail(path, f"expected at most {max_items} items")
            item_schema = schema_node.get("items")
            if isinstance(item_schema, dict):
                for i, item in enumerate(value):
                    ok, err = _validate(item, item_schema, f"{path}[{i}]")
                    if not ok:
                        return ok, err
            return True, None
        if node_type == "string":
            if not isinstance(value, str):
                return _fail(path, "expected string")
            return True, None
        if node_type == "number":
            if not isinstance(value, (int, float)):
                return _fail(path, "expected number")
            minimum = schema_node.get("minimum")
            if isinstance(minimum, (int, float)) and value < minimum:
                return _fail(path, f"expected number >= {minimum}")
            maximum = schema_node.get("maximum")
            if isinstance(maximum, (int, float)) and value > maximum:
                return _fail(path, f"expected number <= {maximum}")
            return True, None
        if node_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return _fail(path, "expected integer")
            return True, None
        if node_type == "boolean":
            if not isinstance(value, bool):
                return _fail(path, "expected boolean")
            return True, None
        return True, None

    return _validate(obj, schema, "$")


_THEME_CORE_SCHEMA = _extract_schema(THEME_SCHEMA)

GOVERNANCE_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "theme_governance",
        "schema": {
            "type": "object",
            "properties": {
                "theme_dict": _THEME_CORE_SCHEMA,
                "change_log": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["merge", "split", "rename", "move"]},
                            "from_ids": {"type": "array", "items": {"type": "string"}},
                            "to_ids": {"type": "array", "items": {"type": "string"}},
                            "reason": {"type": "string"},
                        },
                        "required": ["action", "from_ids", "to_ids", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["theme_dict", "change_log"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def normalize_change_log(change_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for entry in change_log or []:
        from_ids = sorted({str(x) for x in entry.get("from_ids", []) if x})
        to_ids = sorted({str(x) for x in entry.get("to_ids", []) if x})
        normalized.append({
            "action": str(entry.get("action", "")).strip(),
            "from_ids": from_ids,
            "to_ids": to_ids,
            "reason": str(entry.get("reason", "")).strip(),
        })
    normalized.sort(key=lambda e: (e["action"], ",".join(e["from_ids"]), ",".join(e["to_ids"]), e["reason"]))
    return normalized


def _norm_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


def _norm_alpha(label: str) -> str:
    return re.sub(r"[^a-z]+", "", (label or "").lower())


def normalize_theme_dict_order(theme_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not theme_dict or "major_themes" not in theme_dict:
        return theme_dict
    result = copy.deepcopy(theme_dict)
    majors = result.get("major_themes", []) or []

    nonanswer = []
    normal = []
    for major in majors:
        if _norm_alpha(major.get("label", "")) == "nonanswer":
            nonanswer.append(major)
        else:
            normal.append(major)

    normal.sort(key=lambda m: _norm_label(m.get("label", "")))
    nonanswer.sort(key=lambda m: _norm_label(m.get("label", "")))

    ordered = normal + nonanswer
    for major in ordered:
        subs = major.get("subs", []) or []
        subs.sort(key=lambda s: _norm_label(s.get("label", "")))
        major["subs"] = subs

    result["major_themes"] = ordered
    return result


def _index_theme_dict(theme_dict: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    major_by_id: Dict[str, Dict[str, Any]] = {}
    sub_by_id: Dict[str, Dict[str, Any]] = {}
    sub_parent: Dict[str, str] = {}
    for major in theme_dict.get("major_themes", []) or []:
        mid = major.get("id")
        if mid:
            major_by_id[mid] = major
        for sub in major.get("subs", []) or []:
            sid = sub.get("id")
            if sid:
                sub_by_id[sid] = sub
                if mid:
                    sub_parent[sid] = mid
    return major_by_id, sub_by_id, sub_parent


def _find_major_by_id(majors: List[Dict[str, Any]], major_id: str) -> Dict[str, Any] | None:
    for major in majors:
        if major.get("id") == major_id:
            return major
    return None


def _find_sub_by_id(majors: List[Dict[str, Any]], sub_id: str) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    for major in majors:
        for sub in major.get("subs", []) or []:
            if sub.get("id") == sub_id:
                return major, sub
    return None, None


def apply_governance_change_log(
    theme_dict: Dict[str, Any],
    proposed_theme_dict: Dict[str, Any],
    change_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not theme_dict or not change_log:
        return normalize_theme_dict_order(theme_dict)

    updated = copy.deepcopy(theme_dict)
    majors = updated.get("major_themes", []) or []
    updated["major_themes"] = majors

    proposed_major_by_id, proposed_sub_by_id, proposed_sub_parent = _index_theme_dict(proposed_theme_dict)

    def _is_nonanswer_major(major_obj: Dict[str, Any]) -> bool:
        return _norm_alpha(major_obj.get("label", "")) == "nonanswer"

    def _ensure_major(mid: str) -> Dict[str, Any] | None:
        existing = _find_major_by_id(majors, mid)
        if existing:
            return existing
        proposed_major = proposed_major_by_id.get(mid)
        if not proposed_major:
            return None
        majors.append(copy.deepcopy(proposed_major))
        return _find_major_by_id(majors, mid)

    def _update_sub_fields(sub_obj: Dict[str, Any], source_sub: Dict[str, Any]) -> None:
        for key in ("label", "definition", "approx_pct", "examples"):
            if key in source_sub:
                sub_obj[key] = copy.deepcopy(source_sub[key])

    def _update_major_fields(major_obj: Dict[str, Any], source_major: Dict[str, Any]) -> None:
        for key in ("label", "definition", "approx_pct"):
            if key in source_major:
                major_obj[key] = copy.deepcopy(source_major[key])

    for entry in change_log:
        action = entry.get("action")
        from_ids = entry.get("from_ids", []) or []
        to_ids = entry.get("to_ids", []) or []

        if action == "rename":
            if not from_ids:
                continue
            from_id = from_ids[0]
            to_id = to_ids[0] if to_ids else from_id
            major_obj = _find_major_by_id(majors, from_id)
            if major_obj:
                if _is_nonanswer_major(major_obj):
                    continue
                source = proposed_major_by_id.get(to_id) or proposed_major_by_id.get(from_id)
                if source:
                    _update_major_fields(major_obj, source)
                if to_id != from_id:
                    major_obj["id"] = to_id
                continue
            parent_major, sub_obj = _find_sub_by_id(majors, from_id)
            if sub_obj and parent_major and not _is_nonanswer_major(parent_major):
                source = proposed_sub_by_id.get(to_id) or proposed_sub_by_id.get(from_id)
                if source:
                    _update_sub_fields(sub_obj, source)
                if to_id != from_id:
                    sub_obj["id"] = to_id
            continue

        if action == "move":
            if not from_ids or not to_ids:
                continue
            sub_id = from_ids[0]
            target_major_id = to_ids[0]
            parent_major, sub_obj = _find_sub_by_id(majors, sub_id)
            target_major = _ensure_major(target_major_id)
            if not sub_obj or not parent_major or not target_major:
                continue
            if _is_nonanswer_major(parent_major) or _is_nonanswer_major(target_major):
                continue
            parent_major["subs"] = [s for s in parent_major.get("subs", []) if s.get("id") != sub_id]
            target_major.setdefault("subs", []).append(sub_obj)
            continue

        if action == "merge":
            if not from_ids or not to_ids:
                continue
            target_id = to_ids[0]
            source = proposed_sub_by_id.get(target_id)
            target_major_id = proposed_sub_parent.get(target_id)
            target_major = _ensure_major(target_major_id) if target_major_id else None
            if not target_major or _is_nonanswer_major(target_major):
                continue
            _, existing_target = _find_sub_by_id(majors, target_id)
            if not existing_target:
                if not source:
                    continue
                new_sub = copy.deepcopy(source)
                target_major.setdefault("subs", []).append(new_sub)
            elif source:
                _update_sub_fields(existing_target, source)
            for from_id in from_ids:
                if from_id == target_id:
                    continue
                parent_major, _ = _find_sub_by_id(majors, from_id)
                if parent_major and not _is_nonanswer_major(parent_major):
                    parent_major["subs"] = [s for s in parent_major.get("subs", []) if s.get("id") != from_id]
            continue

        if action == "split":
            if not from_ids or not to_ids:
                continue
            from_id = from_ids[0]
            parent_major, _ = _find_sub_by_id(majors, from_id)
            if not parent_major or _is_nonanswer_major(parent_major):
                continue
            parent_major["subs"] = [s for s in parent_major.get("subs", []) if s.get("id") != from_id]
            for new_id in to_ids:
                source = proposed_sub_by_id.get(new_id)
                if not source:
                    continue
                parent_id = proposed_sub_parent.get(new_id) or parent_major.get("id")
                target_major = _ensure_major(parent_id)
                if not target_major or _is_nonanswer_major(target_major):
                    continue
                target_major.setdefault("subs", []).append(copy.deepcopy(source))
            continue

    return normalize_theme_dict_order(updated)

