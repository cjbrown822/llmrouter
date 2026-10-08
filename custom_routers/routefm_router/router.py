"""Frozen RouteFM adapter for LLMRouter's custom-router interface."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter

from custom_routers.routefm_router.context import build_behavioral_context

try:
    import routefm as routefm_package
except ImportError as error:  # pragma: no cover - exercised by plugin discovery
    raise ImportError(
        "RouteFMPluginRouter requires: pip install 'routefm-router[bge]>=1.1,<2'"
    ) from error


class RouteFMPluginRouter(MetaRouter):
    """Route with frozen RouteFM weights and LLMRouter behavioral history."""

    def __init__(self, yaml_path: str):
        super().__init__(model=nn.Identity(), yaml_path=yaml_path)
        if not isinstance(self.llm_data, Mapping) or not self.llm_data:
            raise ValueError("RouteFM requires data_path.llm_data")
        if self.routing_data_train is None:
            raise ValueError("RouteFM requires data_path.routing_data_train")

        hparam = self.cfg.get("hparam", {}) or {}
        self.context_size = int(hparam.get("context_size", 8))
        self.context_seed = int(hparam.get("context_seed", 31010))
        self.device = str(hparam.get("device", "cpu"))
        candidate_models = hparam.get("candidate_models")
        self.context = build_behavioral_context(
            self.routing_data_train,
            self.llm_data,
            context_size=self.context_size,
            seed=self.context_seed,
            candidate_models=candidate_models,
            key_column=hparam.get("context_key"),
            cost_column=hparam.get("cost_column"),
            price_scale=float(hparam.get("price_scale", 1_000_000.0)),
        )
        self.llm_names = list(self.context)
        excluded = [name for name in self.llm_data if name not in self.context]
        if excluded:
            print(
                "[RouteFM] Ignoring candidates without routing observations: "
                + ", ".join(excluded)
            )

        load_options: dict[str, Any] = {
            "encoder": "bge",
            "device": self.device,
            "embedding_batch_size": int(hparam.get("embedding_batch_size", 32)),
            "embedding_max_length": int(hparam.get("embedding_max_length", 512)),
        }
        for option in ("checkpoint", "cache_dir", "embedding_model", "embedding_revision"):
            if hparam.get(option) is not None:
                load_options[option] = hparam[option]
        self.routefm_router = routefm_package.RouteFMRouter.from_pretrained(**load_options)
        self.routefm_router.set_context(self.context)

    @staticmethod
    def _query_text(value: Any) -> str:
        query = value.get("query", "") if isinstance(value, Mapping) else str(value)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return query

    def _format_result(self, value: Any, decision: Any) -> dict[str, Any]:
        output = copy.copy(value) if isinstance(value, Mapping) else {"query": str(value)}
        scores = dict(decision.predicted_scores)
        ordered_scores = sorted(scores.values(), reverse=True)
        output.update({
            "model_name": decision.model_name,
            "predicted_llm": decision.model_name,
            "predicted_llm_name": decision.model_name,
            "method": "routefm",
            "routefm_model_index": decision.model_index,
            "routefm_predicted_scores": scores,
            "routefm_predicted_relative_costs": dict(decision.predicted_relative_costs),
            "routefm_score_margin": (
                ordered_scores[0] - ordered_scores[1]
                if len(ordered_scores) > 1 else 0.0
            ),
            "routefm_context_size": self.context_size,
        })
        return output

    def route_single(self, query_input: dict[str, Any]) -> dict[str, Any]:
        query = self._query_text(query_input)
        decision = self.routefm_router.route(query)
        return self._format_result(query_input, decision)

    def route_batch(self, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if batch is None:
            return []
        if isinstance(batch, (str, bytes)):
            batch = [{"query": str(batch)}]
        values = list(batch)
        queries = [self._query_text(value) for value in values]
        decisions = self.routefm_router.route_batch(queries)
        return [
            self._format_result(value, decision)
            for value, decision in zip(values, decisions, strict=True)
        ]
