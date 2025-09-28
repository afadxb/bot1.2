"""Minimal subset of Pydantic used for configuration models."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Type


class _FieldInfo:
    def __init__(self, default: Any = None, default_factory=None) -> None:
        self.default = default
        self.default_factory = default_factory


def Field(*, default: Any = None, default_factory=None):
    return _FieldInfo(default=default, default_factory=default_factory)


class BaseModel:
    """Very small BaseModel replacement."""

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self.__class__, "__annotations__", {})
        for name, annotation in annotations.items():
            if name in data:
                value = data[name]
            else:
                default = getattr(self.__class__, name, None)
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
