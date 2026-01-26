
from typing import Dict, Any, Tuple, List, Optional, Iterable, Callable, Union
import json
import math
import time

# --- Types ---
ParamKey = Tuple[Tuple[str, Any], ...]  # canonicalized, order-independent key


# --- Canonicalization helpers (robust to lists/tuples/dicts/sets, NumPy scalars) ---
def _to_python_scalar(v):
    """Convert NumPy scalars to native Python scalars; keep others as-is."""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    return v


def _freeze(value: Any) -> Any:
    """
    Make a value hashable & stable for use in keys:
    - scalars -> themselves
    - list/tuple -> tuple of frozen items
    - dict -> tuple of sorted (key, frozen(value)) pairs
    - set -> tuple of sorted frozen items
    """
    value = _to_python_scalar(value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    # basic types: int, float, str, bool, None
    return value


def canonicalize_params(params: Dict[str, Any]) -> ParamKey:
    """
    Deterministic, hashable, order-independent key from a params dict.
    Works even when 'params' is empty: canonicalize_params({}) == ()
    """
    return tuple(sorted((k, _freeze(v)) for k, v in params.items()))


class ResultManager:
    """
    Tracks metrics for each hyperparameter combination (including none).
    
    Key features:
    - **Single add method** (`add_result`) that logs or appends seamlessly.
    - **Batch add** (`add_results`) to record multiple metrics at once.
    - Order-independent param keys (dict order does not matter).
    - Supports repeated runs per metric (stores lists and aggregates on demand).
    - Export to rows/DataFrame; CSV and JSON save/load.
    - Utility methods to query, filter, merge, delete, and summarize.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self,
        allowed_params: Optional[Dict[str, type]] = None,
        auto_timestamp: bool = True,
    ) -> None:
        """
        Create a new ResultStore.

        Args:
            allowed_params: Optional schema for param names/types, e.g.
                            {"alpha": float, "gamma": float, "optimizer": str}
                            If provided, params will be validated on add.
            auto_timestamp: If True, store a 'last_updated' timestamp per config.
        """
        self.store: Dict[ParamKey, Dict[str, Any]] = {}             # ParamKey -> metrics
        self.meta: Dict[ParamKey, Dict[str, Any]] = {}              # ParamKey -> metadata (e.g., last_updated, notes)
        self.allowed_params = allowed_params
        self.auto_timestamp = auto_timestamp

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate against allowed_params schema, if provided."""
        if not self.allowed_params:
            return
        # Check unknown names
        unknown = set(params.keys()) - set(self.allowed_params.keys())
        if unknown:
            raise ValueError(f"Unknown hyperparameter(s): {sorted(unknown)}")
        # Check types (best-effort; allows subclasses)
        for k, v in params.items():
            expected_t = self.allowed_params[k]
            if v is None:
                continue
            try:
                ok = isinstance(v, expected_t)
            except TypeError:
                # If expected_t is typing constructs not usable with isinstance, skip
                ok = True
            if not ok:
                raise TypeError(f"Param '{k}' expects {expected_t}, got {type(v)}")

    def _touch_meta(self, key: ParamKey) -> None:
        """Update metadata for the given key."""
        m = self.meta.setdefault(key, {})
        if self.auto_timestamp:
            m["last_updated"] = time.time()

    # -------------------------------------------------------------------------
    # Adding results
    # -------------------------------------------------------------------------
    def add_result(
        self,
        params: Dict[str, Any],
        metric_name: str,
        value: Any,
    ) -> None:
        """
        Add a single metric result for 'params'.
        Behavior:
          - If the metric doesn't exist yet -> sets as scalar 'value'.
          - If the metric exists as a scalar     -> converts to list and appends.
          - If the metric exists as a list       -> appends to the list.
        This way the caller never needs to know whether it's the first or a repeated result.
        """
        self._validate_params(params)
        key = canonicalize_params(params)
        metrics = self.store.setdefault(key, {})
        current = metrics.get(metric_name)
        if current is None:
            metrics[metric_name] = value
        elif isinstance(current, list):
            current.append(value)
        else:
            metrics[metric_name] = [current, value]
        self._touch_meta(key)

    def add_results(self, params: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Add multiple metrics in one call for the same 'params'.
        Applies the same append logic as 'add_result' per metric.
        """
        for metric_name, value in results.items():
            self.add_result(params, metric_name, value)

    def set_result(self, params: Dict[str, Any], metric_name: str, value: Any) -> None:
        """
        Force-set a metric to a specific scalar/list value (overwrite semantics).
        Useful for corrections or external aggregation.
        """
        self._validate_params(params)
        key = canonicalize_params(params)
        metrics = self.store.setdefault(key, {})
        metrics[metric_name] = value
        self._touch_meta(key)

    def add_note(self, params: Dict[str, Any], note: str) -> None:
        """
        Attach a human-readable note to a configuration (e.g., provenance).
        """
        key = canonicalize_params(params)
        self.meta.setdefault(key, {})
        notes = self.meta[key].setdefault("notes", [])
        notes.append(note)
        self._touch_meta(key)

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------
    def get_result(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return the metrics dict for the given 'params' (or None if absent).
        """
        return self.store.get(canonicalize_params(params))

    def get_meta(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return metadata dict for the given 'params' (or None if absent).
        """
        return self.meta.get(canonicalize_params(params))

    def has_config(self, params: Dict[str, Any]) -> bool:
        """Check if a configuration exists."""
        return canonicalize_params(params) in self.store

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all hyperparameter configurations as dictionaries."""
        return [dict(k) for k in self.store.keys()]

    def list_metrics(self) -> List[str]:
        """List all metric names observed across configurations."""
        names = set()
        for metrics in self.store.values():
            names.update(metrics.keys())
        return sorted(names)

    def items(self) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Iterate over (params_dict, metrics_dict) pairs.
        Note: 'params_dict' is reconstructed from the canonical key.
        """
        for key, metrics in self.store.items():
            yield dict(key), metrics

    def __len__(self) -> int:
        """Number of distinct hyperparameter configurations stored."""
        return len(self.store)

    def __contains__(self, params: Dict[str, Any]) -> bool:
        """Allow `params in rs` checks."""
        return self.has_config(params)

    def __getitem__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allow bracket access: rs[params] -> metrics dict.
        Raises KeyError if absent.
        """
        key = canonicalize_params(params)
        if key not in self.store:
            raise KeyError("Configuration not found")
        return self.store[key]

    # -------------------------------------------------------------------------
    # Deletion and maintenance
    # -------------------------------------------------------------------------
    def remove_config(self, params: Dict[str, Any]) -> None:
        """Delete an entire configuration and its metadata."""
        key = canonicalize_params(params)
        self.store.pop(key, None)
        self.meta.pop(key, None)

    def remove_metric(self, params: Dict[str, Any], metric_name: str) -> None:
        """Delete a specific metric from a configuration."""
        key = canonicalize_params(params)
        metrics = self.store.get(key)
        if metrics and metric_name in metrics:
            del metrics[metric_name]
            self._touch_meta(key)

    def clear(self) -> None:
        """Remove all stored results and metadata."""
        self.store.clear()
        self.meta.clear()

    # -------------------------------------------------------------------------
    # Aggregation and best selection
    # -------------------------------------------------------------------------
    @staticmethod
    def _aggregate(values: Iterable[float], method: str = "mean") -> float:
        """
        Aggregate a sequence of numeric values using a method:
          - 'mean'   : arithmetic average
          - 'median' : middle value (or mean of middle two)
          - 'max'    : maximum
          - 'min'    : minimum
        Returns NaN for empty lists.
        """
        vals = list(values)
        if not vals:
            return math.nan
        if method == "mean":
            return sum(vals) / len(vals)
        if method == "median":
            s = sorted(vals); n = len(s); mid = n // 2
            return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
        if method == "max":
            return max(vals)
        if method == "min":
            return min(vals)
        raise ValueError(f"Unsupported aggregation method: {method}")

    def get_best(
        self,
        metric_name: str,
        mode: str = "max",
        aggregate: Union[str, Callable[[Iterable[float]], float]] = "mean",
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Return (best_params_dict, best_value) for the given 'metric_name'.

        - mode: 'max' or 'min' (choose higher or lower aggregated value)
        - aggregate:
            * String in {'mean','median','max','min'} to aggregate lists, OR
            * Callable(values) -> float for custom aggregation.
          If the metric value is a scalar, aggregation is skipped.
        - filter_fn: optional predicate(params_dict) to restrict candidates.

        Behavior:
          * If a metric value is a list, we aggregate it to a single float.
          * Non-numeric lists will raise if the aggregator needs numeric values.
          * Configurations missing the metric are skipped.
          * Raises KeyError if nothing qualifies.
        """
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")

        # Build aggregator
        if callable(aggregate):
            agg_fn = aggregate
        else:
            agg_method = str(aggregate)
            agg_fn = lambda xs: self._aggregate(xs, method=agg_method)

        best_key = None
        best_value = None

        for key, metrics in self.store.items():
            if metric_name not in metrics:
                continue

            val = metrics[metric_name]
            # Aggregate if list
            if isinstance(val, list):
                # Ensure numeric values for numeric aggregation
                if callable(aggregate):
                    numeric_vals = [float(x) for x in val]
                    val = agg_fn(numeric_vals)
                else:
                    val = self._aggregate([float(x) for x in val], method=agg_method)

            params_dict = dict(key)
            if filter_fn and not filter_fn(params_dict):
                continue

            if best_value is None or (val > best_value if mode == "max" else val < best_value):
                best_value = val
                best_key = key

        if best_key is None:
            raise KeyError(f"No results logged for metric '{metric_name}'")
        return dict(best_key), best_value

    def summarize(
        self,
        metric_name: str,
        aggregate: Union[str, Callable[[Iterable[float]], float]] = "mean",
        sort_mode: str = "desc",
    ) -> List[Tuple[Dict[str, Any], Any]]:
        """
        Return a list of (params_dict, aggregated_value) across all configs for a metric,
        sorted descending (default) or ascending.

        Args:
            metric_name: which metric to summarize
            aggregate: aggregation method or callable
            sort_mode: 'desc' or 'asc'
        """
        items = []
        for key, metrics in self.store.items():
            if metric_name not in metrics:
                continue
            val = metrics[metric_name]
            if isinstance(val, list):
                if callable(aggregate):
                    val = aggregate([float(x) for x in val])
                else:
                    val = self._aggregate([float(x) for x in val], method=str(aggregate))
            items.append((dict(key), val))
        reverse = sort_mode == "desc"
        items.sort(key=lambda kv: kv[1], reverse=reverse)
        return items

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def to_rows(
        self,
        aggregate_lists_as: Optional[Union[str, Callable[[Iterable[float]], float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Flatten the store to row dicts: one row per hyperparameter configuration.
        Each row contains params + metrics.

        - If 'aggregate_lists_as' is None (default), list-valued metrics remain lists.
        - If 'aggregate_lists_as' is a string ('mean','median','max','min')
          or a callable(values)->float, list metrics are aggregated to scalars.
        """
        if callable(aggregate_lists_as):
            agg_fn = aggregate_lists_as
        elif isinstance(aggregate_lists_as, str):
            agg_fn = lambda xs: self._aggregate(xs, method=aggregate_lists_as)
        else:
            agg_fn = None  # keep lists

        rows = []
        for key, metrics in self.store.items():
            row = dict(key)  # params
            for mname, mval in metrics.items():
                if isinstance(mval, list) and agg_fn is not None:
                    try:
                        row[mname] = agg_fn([float(x) for x in mval])
                    except Exception:
                        # If aggregation fails (non-numeric), keep the original list
                        row[mname] = mval
                else:
                    row[mname] = mval
            rows.append(row)
        return rows

    def to_dataframe(
        self,
        aggregate_lists_as: Optional[Union[str, Callable[[Iterable[float]], float]]] = None
    ):
        """
        Convert to a pandas DataFrame (optional dependency).
        - If 'aggregate_lists_as' is provided, list metrics are aggregated to scalars first.
        """
        import pandas as pd
        return pd.DataFrame(self.to_rows(aggregate_lists_as=aggregate_lists_as))

    def save_json(self, path: str) -> None:
        """
        Save results and metadata to JSON. Values must be JSON-serializable.
        Lists remain lists; scalars remain scalars.

        Structure:
            [
              {"params": {...}, "metrics": {...}, "meta": {...}},
              ...
            ]
        """
        serializable = []
        for key, metrics in self.store.items():
            serializable.append({
                "params": dict(key),
                "metrics": metrics,
                "meta": self.meta.get(key, {})
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load_json(cls, path: str):
        """
        Load results from a JSON file previously saved by 'save_json'.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rs = cls()
        for item in data:
            key = canonicalize_params(item["params"])
            rs.store[key] = item["metrics"]
            rs.meta[key] = item.get("meta", {})
        return rs

    def save_csv(
        self,
        path: str,
        aggregate_lists_as: Optional[Union[str, Callable[[Iterable[float]], float]]] = "mean",
    ) -> None:
        """
        Save a CSV table of results. By default, aggregates list-valued metrics to 'mean'.
        """
        import csv
        rows = self.to_rows(aggregate_lists_as=aggregate_lists_as)
        # Collect all columns
        cols = set()
        for r in rows:
            cols.update(r.keys())
        cols = sorted(cols)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # -------------------------------------------------------------------------
    # Merging and combination
    # -------------------------------------------------------------------------
    def merge(
        self,
        other: "ResultManager",
        conflict: str = "append",  # 'append' or 'overwrite'
    ) -> None:
        """
        Merge another ResultStore into this one.

        - conflict='append': existing scalar becomes list; lists are extended.
        - conflict='overwrite': metrics from 'other' replace existing values.
        """
        if conflict not in {"append", "overwrite"}:
            raise ValueError("conflict must be 'append' or 'overwrite'")

        for key, other_metrics in other.store.items():
            my_metrics = self.store.setdefault(key, {})
            for mname, mval in other_metrics.items():
                if conflict == "overwrite":
                    my_metrics[mname] = mval
                else:  # append mode
                    current = my_metrics.get(mname)
                    if current is None:
                        my_metrics[mname] = mval
                    elif isinstance(current, list):
                        if isinstance(mval, list):
                            current.extend(mval)
                        else:
                            current.append(mval)
                    else:
                        # current scalar -> list
                        my_metrics[mname] = [current] + (mval if isinstance(mval, list) else [mval])
            # merge metadata (shallow)
            my_meta = self.meta.setdefault(key, {})
            other_meta = other.meta.get(key, {})
            my_meta.update(other_meta)
            self._touch_meta(key)

    # -------------------------------------------------------------------------
    # Convenience utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def grid(param_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
        """
        Generate all combinations from a param space: {'alpha':[0.1,0.2], 'gamma':[0.9]}
        """
        import itertools
        keys = list(param_space.keys())
        for combo in itertools.product(*(param_space[k] for k in keys)):
            yield dict(zip(keys, combo))

    def __repr__(self) -> str:
        return f"ResultStore(configs={len(self.store)}, metrics={self.list_metrics()})"
