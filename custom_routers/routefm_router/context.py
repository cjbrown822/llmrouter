"""Build deterministic RouteFM behavioral Context from LLMRouter data."""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _observation_cost(
    row: pd.Series,
    metadata: Mapping[str, Any],
    *,
    cost_column: str | None,
    price_scale: float,
) -> float:
    if cost_column is not None:
        value = _finite_number(row[cost_column], cost_column)
        if value < 0:
            raise ValueError(f"{cost_column} must be nonnegative")
        return value

    input_price = _finite_number(metadata.get("input_price", 0.0), "input_price")
    output_price = _finite_number(metadata.get("output_price", 0.0), "output_price")
    if input_price < 0 or output_price < 0:
        raise ValueError("candidate prices must be nonnegative")

    has_split_tokens = "input_tokens" in row.index or "output_tokens" in row.index
    if has_split_tokens:
        input_tokens = (
            0.0 if pd.isna(row.get("input_tokens"))
            else _finite_number(row.get("input_tokens"), "input_tokens")
        )
        output_tokens = (
            0.0 if pd.isna(row.get("output_tokens"))
            else _finite_number(row.get("output_tokens"), "output_tokens")
        )
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be nonnegative")
        return (input_price * input_tokens + output_price * output_tokens) / price_scale

    if "token_num" in row.index and not pd.isna(row.get("token_num")):
        tokens = _finite_number(row.get("token_num"), "token_num")
        if tokens < 0:
            raise ValueError("token_num must be nonnegative")
        # Without a prompt/completion split, the mean price is a documented
        # approximation. Users can point cost_column at an exact observed cost.
        return tokens * (input_price + output_price) / (2.0 * price_scale)

    return 0.0


def build_behavioral_context(
    routing_data: pd.DataFrame,
    llm_data: Mapping[str, Mapping[str, Any]],
    *,
    context_size: int = 8,
    seed: int = 31010,
    candidate_models: Sequence[str] | None = None,
    key_column: str | None = None,
    cost_column: str | None = None,
    price_scale: float = 1_000_000.0,
) -> dict[str, list[dict[str, float | str]]]:
    """Select aligned observations and convert them to RouteFM Context.

    The same deterministic query IDs are selected for every candidate. This
    makes candidate comparisons auditable and prevents row-order differences
    from changing the Context pool.
    """
    if routing_data is None or routing_data.empty:
        raise ValueError("RouteFM requires non-empty routing_data_train")
    if context_size < 1:
        raise ValueError("context_size must be positive")
    if not math.isfinite(price_scale) or price_scale <= 0:
        raise ValueError("price_scale must be positive and finite")

    required = {"model_name", "query", "performance"}
    missing_columns = required - set(routing_data.columns)
    if missing_columns:
        raise ValueError(f"routing_data_train is missing columns: {sorted(missing_columns)}")

    if isinstance(candidate_models, (str, bytes)):
        raise TypeError("candidate_models must be a sequence of model names")
    observed_models = set(routing_data["model_name"].dropna())
    candidates = (
        list(candidate_models)
        if candidate_models is not None
        else [name for name in llm_data if name in observed_models]
    )
    if len(candidates) < 2:
        raise ValueError("RouteFM requires at least two candidate models")
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate_models contains duplicates")
    missing_metadata = [name for name in candidates if name not in llm_data]
    if missing_metadata:
        raise ValueError(f"candidate metadata is missing for: {missing_metadata}")

    if cost_column is not None and cost_column not in routing_data.columns:
        raise ValueError(f"routing_data_train has no cost column {cost_column!r}")

    if key_column is None:
        key_column = (
            "embedding_id"
            if "embedding_id" in routing_data.columns
            and routing_data["embedding_id"].notna().any()
            else "query"
        )
    if key_column not in routing_data.columns:
        raise ValueError(f"routing_data_train has no Context key column {key_column!r}")

    frame = routing_data[routing_data["model_name"].isin(candidates)].copy()
    frame = frame.dropna(subset=[key_column, "query", "performance"])
    present = set(frame["model_name"].unique())
    missing_rows = [name for name in candidates if name not in present]
    if missing_rows:
        raise ValueError(f"routing observations are missing for: {missing_rows}")
    duplicate_counts = frame.groupby([key_column, "model_name"], dropna=False).size()
    if (duplicate_counts > 1).any():
        raise ValueError(
            "routing_data_train contains duplicate rows for a Context key/model pair"
        )

    common_counts = frame.groupby(key_column, sort=False)["model_name"].nunique()
    common_keys = [key for key, count in common_counts.items() if count == len(candidates)]
    if len(common_keys) < context_size:
        raise ValueError(
            f"only {len(common_keys)} aligned Context queries are available for "
            f"{len(candidates)} candidates; context_size={context_size}"
        )
    stable_keys = sorted(common_keys, key=lambda value: (type(value).__name__, str(value)))
    selected_keys = random.Random(seed).sample(stable_keys, context_size)

    result: dict[str, list[dict[str, float | str]]] = {name: [] for name in candidates}
    for key in selected_keys:
        key_rows = frame[frame[key_column] == key]
        if key_rows["query"].nunique(dropna=False) != 1:
            raise ValueError(f"Context key {key!r} maps to multiple query texts")
        for candidate in candidates:
            row = key_rows[key_rows["model_name"] == candidate].iloc[0]
            score = _finite_number(row["performance"], "performance")
            if not 0.0 <= score <= 1.0:
                raise ValueError("performance values must lie in [0,1]")
            cost = _observation_cost(
                row,
                llm_data[candidate],
                cost_column=cost_column,
                price_scale=price_scale,
            )
            result[candidate].append({
                "query": str(row["query"]),
                "score": score,
                "cost": cost,
            })
    return result
