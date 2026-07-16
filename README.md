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

## Next research phase

The next phase must define recommendation ground truth, temporal train/validation/
test splits, retrieval and ranking baselines, repeated runs, and statistical tests
computed from actual predictions. Existing synthetic significance outputs should
not be included in an academic submission.
