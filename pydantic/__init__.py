"""Minimal subset of Pydantic used for configuration models."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type, get_args, get_origin


class _FieldInfo:
    def __init__(self, default: Any = None, default_factory=None, alias: Optional[str] = None) -> None:
        self.default = default
        self.default_factory = default_factory
        self.alias = alias


def Field(*, default: Any = None, default_factory=None, alias: Optional[str] = None):
    return _FieldInfo(default=default, default_factory=default_factory, alias=alias)


class BaseModel:
    """Very small BaseModel replacement."""

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self.__class__, "__annotations__", {})
        for name, annotation in annotations.items():
            field_info = getattr(self.__class__, name, None)
            alias = field_info.alias if isinstance(field_info, _FieldInfo) else None
            lookup_keys = [name]
            if alias:
                lookup_keys.insert(0, alias)
            found = False
            for key in lookup_keys:
                if key in data:
                    value = data[key]
                    found = True
                    break
            if not found:
                default = field_info if isinstance(field_info, _FieldInfo) else getattr(self.__class__, name, None)
                if isinstance(default, _FieldInfo):
                    if default.default_factory is not None:
                        value = default.default_factory()
                    else:
                        value = default.default
                else:
                    value = default
            value = self._coerce(annotation, value)
            setattr(self, name, value)

    @classmethod
    def _coerce(cls, annotation: Any, value: Any) -> Any:
        annotation = cls._resolve_annotation(annotation)
        origin = get_origin(annotation)
        if origin is Optional:
            inner = get_args(annotation)[0]
            if value in (None, ""):
                return None
            return cls._coerce(inner, value)
        if annotation in (int, float, str):
            if value is None:
                return value
            try:
                return annotation(value)
            except Exception:
                return value
        if annotation is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "t", "yes", "y", "on"}:
                    return True
                if normalized in {"0", "false", "f", "no", "n", "off"}:
                    return False
            return bool(value)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if isinstance(value, BaseModel):
                return value
            if isinstance(value, dict):
                return annotation(**value)
        return value

    @classmethod
    def _resolve_annotation(cls, annotation: Any) -> Any:
        if isinstance(annotation, str):
            module = importlib.import_module(cls.__module__)
            try:
                return eval(annotation, module.__dict__)
            except Exception:
                return annotation
        return annotation

    @classmethod
    def model_validate(cls: Type["BaseModel"], data: Dict[str, Any]) -> "BaseModel":
        return cls(**data)

    def model_dump(self) -> Dict[str, Any]:
        result = {}
        annotations = getattr(self.__class__, "__annotations__", {})
        for name in annotations:
            value = getattr(self, name)
            if isinstance(value, BaseModel):
                result[name] = value.model_dump()
            else:
                result[name] = value
        return result

    def copy(self) -> "BaseModel":
        return self.__class__(**self.model_dump())


class BaseSettings(BaseModel):
    class Config:
        env_file: Optional[str] = None
        env_file_encoding: str = "utf-8"
        case_sensitive: bool = False

    def __init__(self, **data: Any) -> None:
        env_data = self._load_env_file()
        env_data.update(self._load_env_vars())
        env_data.update(data)
        super().__init__(**env_data)

    @classmethod
    def _aliases(cls) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        annotations = getattr(cls, "__annotations__", {})
        for name in annotations:
            field_info = getattr(cls, name, None)
            alias = field_info.alias if isinstance(field_info, _FieldInfo) else None
            mapping[(alias or name)] = name
        return mapping

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        case_sensitive = getattr(getattr(cls, "Config", object), "case_sensitive", False)
        return key if case_sensitive else key.upper()

    @classmethod
    def _load_env_vars(cls) -> Dict[str, Any]:
        mapping = cls._aliases()
        result: Dict[str, Any] = {}
        env = os.environ
        for alias, field in mapping.items():
            keys: Iterable[str]
            if getattr(getattr(cls, "Config", object), "case_sensitive", False):
                keys = (alias,)
            else:
                keys = {alias, alias.upper()}
            for candidate in keys:
                if candidate in env:
                    result[field] = env[candidate]
                    break
        return result

    @classmethod
    def _load_env_file(cls) -> Dict[str, Any]:
        config = getattr(cls, "Config", None)
        path = getattr(config, "env_file", None)
        if not path:
            return {}
        file_path = Path(path)
        if not file_path.exists():
            return {}
        encoding = getattr(config, "env_file_encoding", "utf-8")
        mapping = cls._aliases()
        result: Dict[str, Any] = {}
        for line in file_path.read_text(encoding=encoding).splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            norm = cls._normalize_key(key)
            for alias, field in mapping.items():
                if cls._normalize_key(alias) == norm and field not in result:
                    result[field] = value
                    break
        return result
