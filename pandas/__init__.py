"""A minimal pandas-like shim used for testing without external dependencies."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def _is_nan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class Series:
    """A minimal Series implementation covering required operations."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: Any = None,
        index: Optional[Sequence[Any]] = None,
        dtype: Any = None,
        name: Optional[str] = None,
    ) -> None:
        if isinstance(data, Series):
            self._data = data._data.copy()
            self._index = data._index.copy()
        elif isinstance(data, (list, tuple, np.ndarray)):
            arr = list(data)
            self._data = arr
            if index is None:
                self._index = list(range(len(arr)))
            else:
                self._index = list(index)
        elif index is not None:
            self._data = [data for _ in index]
            self._index = list(index)
        else:
            self._data = [data] if data is not None else []
            self._index = [0] if self._data else []
        self.name = name
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._data[key]
        if key in self._index:
            pos = self._index.index(key)
            return self._data[pos]
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, int):
            self._data[key] = value
            return
        if key in self._index:
            pos = self._index.index(key)
            self._data[pos] = value
        else:
            self._index.append(key)
            self._data.append(value)

    def __add__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a / b)

    def __ge__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a >= b)

    def __le__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a <= b)

    def __gt__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a > b)

    def __lt__(self, other: Any) -> "Series":
        return self._comparison_op(other, lambda a, b: a < b)

    def __eq__(self, other: Any) -> "Series":  # type: ignore[override]
        return self._comparison_op(other, lambda a, b: a == b)

    def __ne__(self, other: Any) -> "Series":  # type: ignore[override]
        return self._comparison_op(other, lambda a, b: a != b)

    def _binary_op(self, other: Any, op: Callable[[Any, Any], Any]) -> "Series":
        if isinstance(other, Series):
            data = [op(a, b) for a, b in zip(self._data, other._data)]
        else:
            data = [op(a, other) for a in self._data]
        return Series(data, index=self._index)

    def _comparison_op(self, other: Any, op: Callable[[Any, Any], bool]) -> "Series":
        if isinstance(other, Series):
            data = [op(a, b) for a, b in zip(self._data, other._data)]
        else:
            data = [op(a, other) for a in self._data]
        return Series(data, index=self._index)

    def map(self, func: Callable[[Any], Any]) -> "Series":
        return Series([func(item) for item in self._data], index=self._index)

    def apply(self, func: Callable[[Any], Any]) -> "Series":
        return self.map(func)

    def fillna(self, value: Any) -> "Series":
        return Series([value if _is_nan(item) else item for item in self._data], index=self._index)

    def astype(self, typ: Any) -> "Series":
        return Series([typ(item) if not _is_nan(item) else item for item in self._data], index=self._index)

    def dropna(self) -> "Series":
        data = [item for item in self._data if not _is_nan(item)]
        index = [idx for idx, item in zip(self._index, self._data) if not _is_nan(item)]
        return Series(data, index=index)

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def clip(self, lower: Optional[float] = None, upper: Optional[float] = None) -> "Series":
        def _clip(value: Any) -> Any:
            if _is_nan(value):
                return value
            if lower is not None and value < lower:
                value = lower
            if upper is not None and value > upper:
                value = upper
            return value

        return Series([_clip(item) for item in self._data], index=self._index)

    def replace(self, old: Any, new: Any) -> "Series":
        return Series([new if item == old else item for item in self._data], index=self._index)

    def isna(self) -> "Series":
        return Series([_is_nan(item) for item in self._data], index=self._index)

    def notna(self) -> "Series":
        return Series([not flag for flag in self.isna()._data], index=self._index)

    def all(self) -> bool:
        return all(bool(item) for item in self._data)

    def any(self) -> bool:
        return any(bool(item) for item in self._data)

    def sum(self) -> Any:
        values = [item for item in self._data if not _is_nan(item)]
        return sum(values)

    def min(self) -> Any:
        values = [item for item in self._data if not _is_nan(item)]
        return min(values) if values else None

    def max(self) -> Any:
        values = [item for item in self._data if not _is_nan(item)]
        return max(values) if values else None

    def to_list(self) -> List[Any]:
        return list(self._data)

    def tolist(self) -> List[Any]:
        return self.to_list()

    def to_dict(self) -> Dict[Any, Any]:
        return {idx: value for idx, value in zip(self._index, self._data)}

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def values(self) -> List[Any]:
        return self._data

    @property
    def index(self) -> List[Any]:
        return self._index

    class _ILoc:
        def __init__(self, series: "Series") -> None:
            self.series = series

        def __getitem__(self, key: int) -> Any:
            return self.series._data[key]

    @property
    def iloc(self) -> "Series._ILoc":
        return Series._ILoc(self)

    def value_counts(self) -> "Series":
        counts: Dict[Any, int] = {}
        for item in self._data:
            counts[item] = counts.get(item, 0) + 1
        keys = list(counts.keys())
        values = [counts[key] for key in keys]
        return Series(values, index=keys)

    def __array__(self) -> np.ndarray:
        return np.array(self._data, dtype=object)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Series({self._data})"


class DataFrame:
    """A minimal DataFrame implementation for the project."""

    def __init__(self, data: Any = None, columns: Optional[Sequence[str]] = None) -> None:
        if data is None:
            self._columns: List[str] = []
            self._data: Dict[str, Series] = {}
            self._index: List[int] = []
            return

        if isinstance(data, list):
            rows: List[Dict[str, Any]]
            if not data:
                # Preserve explicit column ordering when provided; otherwise start empty.
                self._columns = list(columns or [])
                self._data = {col: Series([]) for col in self._columns}
                self._index = []
                return
            if isinstance(data[0], dict):
                rows = data
                cols = columns or list(rows[0].keys())
                all_cols: List[str] = list(
                    dict.fromkeys([col for row in rows for col in row.keys()])
                )
                if columns is None:
                    columns = all_cols
                values: Dict[str, List[Any]] = {col: [] for col in columns}
                for row in rows:
                    for col in columns:
                        values[col].append(row.get(col))
                self._columns = list(columns)
                self._data = {col: Series(values[col]) for col in self._columns}
                self._index = list(range(len(rows)))
            else:
                raise TypeError("Unsupported data format for DataFrame")
        elif isinstance(data, dict):
            if columns is None:
                columns = list(data.keys())
            length = None
            for col in columns:
                seq = list(data.get(col, []))
                if length is None:
                    length = len(seq)
                elif len(seq) != length:
                    raise ValueError("Column lengths must match")
            self._columns = list(columns)
            self._data = {col: Series(list(data.get(col, []))) for col in self._columns}
            self._index = list(range(length or 0))
        else:
            raise TypeError("Unsupported data format for DataFrame")

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @property
    def index(self) -> List[Any]:
        return list(self._index)

    def copy(self) -> "DataFrame":
        new = DataFrame()
        new._columns = self.columns
        new._index = self.index
        new._data = {col: Series(series) for col, series in self._data.items()}
        return new

    def rename(self, columns: Dict[str, str]) -> "DataFrame":
        data = {columns.get(col, col): Series(series) for col, series in self._data.items()}
        df = DataFrame()
        df._columns = list(data.keys())
        df._index = self.index
        df._data = data
        return df

    def __len__(self) -> int:
        return len(self._index)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    def __getitem__(self, column: Any) -> Any:
        if isinstance(column, list):
            data = {col: Series(self._data[col]) for col in column}
            df = DataFrame()
            df._columns = column
            df._index = self.index
            df._data = data
            return df
        return self._data[column]

    def __setitem__(self, column: str, values: Any) -> None:
        if isinstance(values, Series):
            series = Series(values)
        elif isinstance(values, list):
            series = Series(values, index=self._index)
        else:
            series = Series(values, index=self._index)
        if not self._index:
            self._index = series.index
        self._data[column] = series
        if column not in self._columns:
            self._columns.append(column)

    def get(self, column: str, default: Any = None) -> Any:
        return self._data.get(column, default)

    def iterrows(self) -> Iterator[Tuple[Any, Series]]:
        for idx_pos, idx in enumerate(self._index):
            values = [self._data[col]._data[idx_pos] for col in self._columns]
            row = Series(values, index=self._columns)
            yield idx, row

    def apply(self, func: Callable[[Dict[str, Any]], Any], axis: int = 0) -> Series:
        if axis != 1:
            raise NotImplementedError("Only axis=1 supported in this shim")
        results = []
        for _, row in self.iterrows():
            results.append(func(row))
        return Series(results, index=self._index)

    def sort_values(
        self,
        by: Sequence[str],
        ascending: Sequence[bool] | bool = True,
        inplace: bool = False,
    ) -> Optional["DataFrame"]:
        if isinstance(by, str):
            by = [by]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)

        def sort_key(pos: int) -> Tuple:
            return tuple(
                self._data[col]._data[pos] if asc else _sort_neg(self._data[col]._data[pos])
                for col, asc in zip(by, ascending)
            )

        def _sort_neg(value: Any) -> Any:
            if isinstance(value, (int, float)) and not _is_nan(value):
                return -value
            return value

        order = sorted(range(len(self._index)), key=sort_key)
        if not all(ascending):
            # For descending columns we already negated numeric values; maintain order
            pass
        new_data = {col: [series._data[i] for i in order] for col, series in self._data.items()}
        new_index = [self._index[i] for i in order]
        target = self if inplace else DataFrame()
        target._columns = self.columns
        target._index = new_index
        target._data = {col: Series(values, index=new_index) for col, values in new_data.items()}
        if inplace:
            return None
        return target

    def head(self, n: int) -> "DataFrame":
        indices = self._index[:n]
        data = {col: series._data[:n] for col, series in self._data.items()}
        df = DataFrame()
        df._columns = self.columns
        df._index = indices
        df._data = {col: Series(values, index=indices) for col, values in data.items()}
        return df

    class _Loc:
        def __init__(self, df: "DataFrame") -> None:
            self.df = df

        def __getitem__(self, key: Tuple[Any, str]) -> Any:
            if isinstance(key, list):
                return self.df._slice_rows(key)
            if not isinstance(key, tuple):
                return self.df._slice_rows([key])
            row_key, col_key = key
            if row_key in self.df._index:
                pos = self.df._index.index(row_key)
            elif isinstance(row_key, int):
                pos = row_key
            else:
                raise KeyError(row_key)
            return self.df._data[col_key]._data[pos]

    @property
    def loc(self) -> "DataFrame._Loc":
        return DataFrame._Loc(self)

    class _ILoc:
        def __init__(self, df: "DataFrame") -> None:
            self.df = df

        def __getitem__(self, key: int) -> Series:
            idx = self.df._index[key]
            values = [self.df._data[col]._data[key] for col in self.df._columns]
            return Series(values, index=self.df._columns)

    @property
    def iloc(self) -> "DataFrame._ILoc":
        return DataFrame._ILoc(self)

    def _slice_rows(self, keys: List[Any]) -> "DataFrame":
        positions = []
        for key in keys:
            if isinstance(key, int):
                positions.append(key)
            elif key in self._index:
                positions.append(self._index.index(key))
            else:
                raise KeyError(key)
        new_index = [self._index[pos] for pos in positions]
        data = {col: [series._data[pos] for pos in positions] for col, series in self._data.items()}
        df = DataFrame()
        df._columns = self.columns
        df._index = new_index
        df._data = {col: Series(values, index=new_index) for col, values in data.items()}
        return df

    def to_dict(self, orient: str = "records") -> List[Dict[str, Any]]:
        if orient != "records":
            raise NotImplementedError("Only records orient supported")
        records = []
        for idx_pos in range(len(self._index)):
            record = {col: self._data[col]._data[idx_pos] for col in self._columns}
            records.append(record)
        return records

    def value_counts(self) -> Series:
        raise NotImplementedError

    def to_csv(self, path: Path | str, index: bool = False) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if index:
                writer.writerow(["index", *self._columns])
                for idx_pos, idx in enumerate(self._index):
                    writer.writerow([idx, *[self._data[col]._data[idx_pos] for col in self._columns]])
            else:
                writer.writerow(self._columns)
                for idx_pos in range(len(self._index)):
                    writer.writerow([self._data[col]._data[idx_pos] for col in self._columns])

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DataFrame(columns={self._columns}, rows={len(self._index)})"


def SeriesConstructor(data: Any = None, index: Optional[Sequence[Any]] = None, dtype: Any = None, name: Optional[str] = None) -> Series:
    return Series(data=data, index=index, dtype=dtype, name=name)


Series = Series  # type: ignore


def to_numeric(data: Any, errors: str = "raise") -> Series:
    series = Series(data)
    converted = []
    for item in series:
        if _is_nan(item):
            converted.append(np.nan)
            continue
        try:
            converted.append(float(item))
        except (TypeError, ValueError):
            if errors == "coerce":
                converted.append(np.nan)
            else:
                raise
    return Series(converted, index=series.index)


def read_csv(path: Path | str) -> DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    return DataFrame(rows)


def DataFrameConstructor(data: Any = None, columns: Optional[Sequence[str]] = None) -> DataFrame:
    return DataFrame(data=data, columns=columns)


DataFrame = DataFrame  # type: ignore


def isna(value: Any) -> bool:
    return _is_nan(value)


def notna(value: Any) -> bool:
    return not _is_nan(value)
