"""Minimal YAML loader sufficient for strategy config."""

from __future__ import annotations

from typing import Any, List


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token in {"true", "True"}:
        return True
    if token in {"false", "False"}:
        return False
    if token in {"null", "None", "~"}:
        return None
    if token.startswith("\"") and token.endswith("\""):
        return token[1:-1]
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _parse_list(lines: List[str], indent: int) -> List[Any]:
    items: List[Any] = []
    while lines:
        line = lines[0]
        if not line.strip():
            lines.pop(0)
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent < indent or not line.strip().startswith("-"):
            break
        lines.pop(0)
        value = line.strip()[1:].strip()
        if value:
            items.append(_parse_scalar(value))
        else:
            items.append(_parse_mapping(lines, indent + 2))
    return items


def _parse_mapping(lines: List[str], indent: int = 0) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    while lines:
        if not lines:
            break
        line = lines[0]
        stripped = line.strip()
        if not stripped:
            lines.pop(0)
            continue
        current_indent = len(line) - len(stripped)
        if current_indent < indent:
            break
        if ":" not in stripped:
            break
        lines.pop(0)
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            if lines and lines[0].strip().startswith("-"):
                mapping[key] = _parse_list(lines, indent + 2)
            else:
                mapping[key] = _parse_mapping(lines, indent + 2)
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                mapping[key] = []
            else:
                mapping[key] = [_parse_scalar(item) for item in inner.split(",")]
        else:
            mapping[key] = _parse_scalar(value)
    return mapping


def safe_load(text: str) -> Any:
    lines = text.splitlines()
    return _parse_mapping(lines)
