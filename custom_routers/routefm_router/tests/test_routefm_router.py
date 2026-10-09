from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import pandas as pd
import yaml

from custom_routers.routefm_router.context import build_behavioral_context
from custom_routers.routefm_router.router import RouteFMPluginRouter


def _routing_frame() -> pd.DataFrame:
    rows = []
    for key, query in ((1, "easy"), (2, "hard"), (3, "shared")):
        rows.extend([
            {
                "embedding_id": key,
                "query": query,
                "model_name": "small",
                "performance": key / 3,
                "input_tokens": 100,
                "output_tokens": 10,
            },
            {
                "embedding_id": key,
                "query": query,
                "model_name": "large",
                "performance": 1.0,
                "input_tokens": 100,
                "output_tokens": 20,
            },
        ])
    return pd.DataFrame(rows)


def _llm_data() -> dict:
    return {
        "small": {"input_price": 0.2, "output_price": 0.2},
        "large": {"input_price": 1.0, "output_price": 2.0},
    }


class BehavioralContextTest(unittest.TestCase):
    def test_context_is_aligned_deterministic_and_costed(self) -> None:
        first = build_behavioral_context(
            _routing_frame(), _llm_data(), context_size=2, seed=7
        )
        second = build_behavioral_context(
            _routing_frame().sample(frac=1, random_state=9),
            _llm_data(),
            context_size=2,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [item["query"] for item in first["small"]],
            [item["query"] for item in first["large"]],
        )
        self.assertAlmostEqual(first["small"][0]["cost"], 22 / 1_000_000)
        self.assertAlmostEqual(first["large"][0]["cost"], 140 / 1_000_000)

    def test_context_requires_every_candidate(self) -> None:
        frame = _routing_frame().query("model_name == 'small'")
        with self.assertRaisesRegex(ValueError, "missing"):
            build_behavioral_context(
                frame,
                _llm_data(),
                context_size=1,
                candidate_models=["small", "large"],
            )


class _FakeRouteFM:
    def __init__(self) -> None:
        self.context = None

    def set_context(self, context):
        self.context = context
        return self

    def route(self, query):
        return self.route_batch([query])[0]

    def route_batch(self, queries):
        return [SimpleNamespace(
            model_name="large",
            model_index=1,
            predicted_scores={"small": 0.25, "large": 0.75},
            predicted_relative_costs={"small": 0.0, "large": 1.0},
        ) for _ in queries]


class RouteFMPluginRouterTest(unittest.TestCase):
    def test_standard_router_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            routing_path = root / "routing.jsonl"
            with routing_path.open("w", encoding="utf-8") as handle:
                for row in _routing_frame().to_dict(orient="records"):
                    handle.write(json.dumps(row) + "\n")
            llm_path = root / "llm.json"
            llm_path.write_text(json.dumps(_llm_data()), encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "data_path": {
                    "routing_data_train": str(routing_path),
                    "llm_data": str(llm_path),
                },
                "hparam": {"context_size": 2, "context_seed": 7},
            }), encoding="utf-8")

            fake = _FakeRouteFM()
            with mock.patch.object(
                __import__(
                    "custom_routers.routefm_router.router",
                    fromlist=["routefm_package"],
                ).routefm_package.RouteFMRouter,
                "from_pretrained",
                return_value=fake,
            ) as load:
                router = RouteFMPluginRouter(str(config_path))

            load.assert_called_once_with(
                encoder="bge",
                device="cpu",
                embedding_batch_size=32,
                embedding_max_length=512,
            )
            self.assertEqual(list(fake.context), ["small", "large"])
            result = router.route_single({"query": "new query", "id": 4})
            self.assertEqual(result["model_name"], "large")
            self.assertEqual(result["predicted_llm"], "large")
            self.assertEqual(result["method"], "routefm")
            self.assertEqual(result["id"], 4)
            self.assertAlmostEqual(result["routefm_score_margin"], 0.5)


if __name__ == "__main__":
    unittest.main()
