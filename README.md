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

## Versioned build workflow

The source CSV is immutable. Topic assignments, author mappings, graph data,
model checkpoints, and vector indexes are stored as separate v2 artifacts.

```bash
# Expensive: generates topics when v2 topic artifacts do not exist,
# builds the graph, and trains the GNN.
python scripts/train_pipeline.py

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
