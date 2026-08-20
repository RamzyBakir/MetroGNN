# MetroGNN

MetroGNN is a research prototype for scholarly paper recommendation. It combines
semantic paper retrieval with author and community signals from a heterogeneous
DBLP graph and exposes the pipeline through MCP tools.

## Current research status

The repository is undergoing a staged reproducibility and evaluation redesign.
The v2 artifact pipeline fixes stable author identity, topic persistence, graph
citation relations, artifact compatibility, and MCP contracts. The current
heuristic training targets and historical statistical reports have **not** yet
been replaced; they must not be treated as final paper evidence.

The files under `agents/prompts/` are design material. The current runtime is an
MCP tool server and deterministic strategy pipeline, not an evaluated LLM agent.

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

All active paths are configured in `config.yaml`. Generated data, models, vector
indexes, and results are intentionally ignored by Git.

## Docker quick start

```bash
docker compose build
docker compose run --rm metrognn
```

On a new machine, the default Docker run validates the runtime artifacts. If
`models` is empty, it builds the graph, topics, checkpoint, and vector store from
the bundled CSV at `train/DBLP-Citation-network-V18-1-20-25_clean.csv`, then runs
the demo inference query. The bootstrap/training path uses seed `42` by default.

For an NVIDIA GPU workstation, use the separate GPU Compose file:

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm metrognn
```

See `DOCKER.md` for GPU checks and `--device auto|cpu|cuda` examples.

## Versioned build workflow

The source CSV is immutable. Topic assignments, author mappings, graph data,
model checkpoints, and vector indexes are stored as separate v2 artifacts.

```bash
# Expensive: generates topics when v2 topic artifacts do not exist,
# builds the graph, and trains the GNN.
python scripts/train_pipeline.py

# One-command artifact validation/regeneration for Docker or local demos.
python scripts/bootstrap_artifacts.py

# Explicitly replace an existing topic artifact family.
python scripts/train_pipeline.py --force-topics

# Build the semantic index only after a compatible pipeline manifest exists.
python -m gnn_pipeline.data.vector_store ingest

# Run a recommendation query.
python -m gnn_pipeline.inference --query "graph neural networks"

# Start the MCP server over stdio.
python -m mcp_server
```

Existing legacy artifacts are not loaded as v2 artifacts. Runtime loading checks
dataset and artifact checksums plus a pipeline ID and fails rather than silently
mixing incompatible graph, model, topic, author, or vector-store files.

## Tests

The lightweight suite does not download embedding models, rebuild full artifacts,
or train on DBLP:

```bash
python scripts/test_pipeline.py
# or, when pytest is installed
pytest
```

## GPU Acceleration & Full Results Generation

MetroGNN supports complete CUDA acceleration across model training, heterogeneous message passing, embedding metrics, and multi-panel visualizations:

```bash
# Train the multi-task heterogeneous GNN on GPU from scratch:
docker compose -f docker-compose.gpu.yml run --rm metrognn python scripts/train_pipeline.py --device cuda

# Generate all evaluation reports, LaTeX tables, and publication visualizations:
docker compose -f docker-compose.gpu.yml run --rm metrognn python scripts/generate_all_results.py --device cuda
```

Key features:
- **Fast Heterogeneous Message Passing**: 4-layer HeteroConv over paper, author, and topic subgraphs.
- **CUDA Tensor Distances**: Tensor-accelerated silhouette score and separation ratio computed in <10ms.
- **Publication Figures**: 10k-node t-SNE with centroid badges, 2-panel force-directed community graphs with diagnostics cards, and polar capability radar charts.

## V2 graph schema

Node types:

- `paper`
- `author`, keyed by DBLP ID or a conservative normalized name+organization fallback
- `topic`

Relations:

- `author -> authored -> paper` and reverse
- `author -> collaborates_with -> author`
- `paper -> has_topic -> topic` and reverse
- `paper -> cites -> paper` and `paper -> cited_by -> paper`

Only references to papers present in the configured subset become citation edges.
Coverage is recorded in the pipeline manifest.

## Temporal recommendation evaluation

The primary offline task is prior-work recommendation: a query paper's title,
abstract, and keywords are used to rank papers published in strictly earlier
years. Internal citations provide incomplete binary qrels; uncited candidates are
unjudged rather than confirmed irrelevant.

The main protocol uses 2021-2022 for training queries, 2023 for validation, and
2024 for the final test. Snapshot citation counts are excluded from temporal
popularity signals. Build the versioned benchmark with:

```bash
python scripts/build_recommendation_benchmark.py
```

Run configured full-corpus baselines with:

```bash
python scripts/evaluate_recommenders.py \
  --proposed seeded_ppr --baseline tfidf
```

Both scripts default to `configs/experiments/citation_temporal_v1.yaml`, which
records the benchmark years, enabled baselines, random seeds, metrics, and
statistical settings.

The bundled local CSV is a legacy metadata-only extract with no per-paper
reference lists. The default config therefore opts into a clearly labeled
`legacy_metadata_citation_proxy` pilot benchmark so the evaluation stack can be
smoke-tested end to end. Use real reference lists, not this proxy output, for
paper evidence.

The default non-neural methods are random, historical citation popularity,
recency, TF-IDF, BM25, and semantic-seeded Personalized PageRank. MiniLM and a
scholarly embedding adapter are implemented but disabled by default to prevent
accidental expensive encoding. Every ranked item, seed, latency, per-query
metric, bootstrap interval, and paired comparison is persisted.

Learned systems should use at least five seeds. Confidence intervals bootstrap
actual matched queries after averaging seeds; paired randomization and Wilcoxon
tests operate on observed per-query differences with Holm correction.

The sparse citation benchmark is suitable as an automated pilot but should be
supplemented by an expanded citation corpus or human graded relevance judgments
for the paper. Existing synthetic significance outputs should not be included in
an academic submission.

## MCP/agent trace evaluation

The repository does not yet implement a live LLM agent. To evaluate one later,
save tool-call traces and score them against the fixed tasks in
`agents/benchmarks/mcp_agent_tasks.jsonl`:

```bash
python scripts/evaluate_mcp_agent_traces.py --traces saved_results/agent_traces.jsonl
```

Trace files are JSONL with one record per benchmark task. Each record must
include `task_id`, `tool_calls`, and `final_response`; full benchmark coverage is
required unless `--allow-partial` is passed for debugging.

The evaluator reports tool selection, argument accuracy, valid tool sequence
rate, failure recovery, evidence grounding, and an aggregate score.
For blinded human review, use
`agents/benchmarks/mcp_agent_human_rubric.json` to rate usefulness, grounding,
actionability, personalization, and factuality.

## Profiling

For local timing diagnostics after artifacts exist:

```bash
python scripts/profile_training.py --steps 20
python scripts/profile_serving.py --runs 20 --query "graph neural networks"
```

These write JSON profiles with artifact validation/load times, target
construction and training-step latency, warm GNN forward latency, and optional
end-to-end query latency.

## Phase 3 model experiments

Phase 3 uses frozen MiniLM or scholarly SPECTER text embeddings, temporal graph
snapshots, deterministic positive/unjudged pair sampling, feature rankers, a text
dual encoder, citation GraphSAGE, heterogeneous GraphSAGE, and HGT. Expensive
steps are always explicit:

```bash
# Build one frozen embedding cache (downloads/encodes only when invoked).
python scripts/build_recommendation_embeddings.py --embedding minilm

# Build training and validation snapshots; the 2023 final-test cutoff is omitted.
python scripts/build_temporal_graphs.py --embedding minilm --relations full

# Build reproducible random and TF-IDF hard-negative pairs.
python scripts/prepare_recommendation_samples.py --embedding minilm

# Evaluate frozen retrieval on validation only.
python scripts/evaluate_embedding_retrieval.py --embedding minilm

# Tune one model family on validation only, then freeze the five-seed winner.
python scripts/tune_recommendation_model.py \
  --embedding minilm --architecture hetero_sage --relations full
```

The active v2 protocol uses residual text/graph transformations initialized to
exactly preserve the frozen MiniLM geometry. Epoch 0 is evaluated and eligible
for selection. A predeclared neural fusion weight in `[0, 0.25, 0.5, 0.75, 1]`
is selected on validation; zero exactly recovers TF-IDF, preventing destructive
random projections from being mistaken for useful learning. The original v1
configuration and outputs remain preserved.

The coarse search is capped at 12 configurations. Only its top two validation
configurations are repeated over the five declared seeds. Selection uses
validation nDCG@10, with Recall@20 as a tie-breaker.

Relation and feature ablations are predeclared in
`configs/experiments/phase3_model_validation_v2.yaml` and run only on validation.
Graph snapshots use cutoffs 2020/2021 for training and 2022 for validation. Topic
clusters are fitted only on the training partition. Snapshot citation edges never
contain the query-year positive links.

Final evaluation is deliberately separate. Both frozen retrieval caches must be
built so MiniLM and SPECTER are evaluated in the same one-time event. After a
model is frozen, build the 2023 graph cutoff explicitly with
`--include-final-test`, then invoke
`run_final_recommendation_test.py --final-test ...`, for example:

```bash
python scripts/run_final_recommendation_test.py --final-test \
  --frozen-selection saved_results/phase3_validation_v2/minilm/hetero_sage/full/frozen_selection.json \
  --embedding minilm --architecture hetero_sage --relations full
```

A registry binds the frozen selection, five checkpoint hashes, benchmark,
embedding/architecture context, both frozen retrieval caches, and the existing
baseline run. It refuses a second final-test execution for the same experiment
ID.
