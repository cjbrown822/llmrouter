# RouteFM Router

**Type:** pretrained, frozen, in-context router. No target-domain router
training is performed.

[RouteFM](https://github.com/LAMDA-Model-Reuse/RouteFM) characterizes anonymous
candidate models from a small behavioral Context and predicts their quality on
each new query. This plugin adapts LLMRouter's historical routing rows into
that Context and implements the standard `route_single` / `route_batch`
contract.

Paper: [Pretrain Once, Route Anywhere: Towards a Foundation Model for LLM
Routing](https://arxiv.org/abs/2609.37362).

## Install

From the LLMRouter repository:

```bash
python -m pip install -e ".[routefm]"
```

This installs `routefm-router[bge]`. The released RouteFM-BGE checkpoint and
the compatible BGE encoder revision are downloaded from Hugging Face on first
use and then reused from the normal Hugging Face cache.

## Run

```bash
llmrouter infer --router routefm_router \
  --config custom_routers/routefm_router/config.yaml \
  --query "Prove that there are infinitely many primes." \
  --route-only
```

The result contains the usual `model_name`, `predicted_llm`, and
`predicted_llm_name` fields plus RouteFM's per-candidate quality and relative
cost predictions:

```text
routefm_predicted_scores
routefm_predicted_relative_costs
routefm_score_margin
routefm_context_size
```

The default decision rule selects maximum predicted quality. Relative cost is
reported but is not used for selection.

## How Context is constructed

At initialization the adapter:

1. reads `routing_data_train` and `llm_data` through LLMRouter's standard
   `MetaRouter` loader;
2. finds query IDs observed for every configured candidate;
3. deterministically samples `context_size` aligned queries with
   `context_seed`;
4. converts `performance` to the RouteFM quality signal;
5. computes observed dollar cost from input/output token counts and candidate
   prices; and
6. re-embeds the selected query text with `BAAI/bge-base-en-v1.5` before loading
   the frozen RouteFM-BGE router.

LLMRouter's bundled `query_embeddings_longformer.pt` is intentionally not
used. It is also 768-dimensional, but it belongs to a different representation
space and is incompatible with the released RouteFM-BGE checkpoint.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `context_size` | `8` | Observations per candidate. |
| `context_seed` | `31010` | Deterministic query selection seed. |
| `context_key` | auto | Shared query ID column; prefers `embedding_id`. |
| `candidate_models` | metadata/history intersection | Optional candidate subset and order. |
| `device` | `cpu` | Device for RouteFM and BGE. |
| `cost_column` | unset | Exact nonnegative observed cost column, if available. |
| `price_scale` | `1000000` | Denominator for token-price metadata. |
| `checkpoint` | released BGE weight | Optional local RouteFM checkpoint. |
| `cache_dir` | HF default | Optional Hugging Face cache directory. |

If `cost_column` is unset, the adapter uses `input_tokens`, `output_tokens`,
`input_price`, and `output_price`. If only `token_num` is available, it uses
the mean input/output price as an explicit approximation. With no token counts,
cost defaults to zero for all candidates.

RouteFM assumes scores are comparable and lie in `[0,1]`, every candidate has
at least one observation, and Context queries do not leak held-out evaluation
targets. The adapter fails with a descriptive error when these requirements
are not met.

By default, candidates must occur in both `llm_data` (so LLMRouter can execute
them) and `routing_data_train` (so RouteFM can characterize them). Candidates
missing either side are excluded with a startup message. An explicit
`candidate_models` list is validated strictly instead.
