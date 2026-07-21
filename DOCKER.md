## Requirements

- Docker Desktop or Docker Engine must be running.
- Run commands from the project root, the folder that contains `Dockerfile`,
  `docker-compose.yml`, and `config.yaml`.
- The local folders `train`, `models`, and `saved_results` are used by Docker
  Compose as runtime mounts.

## Build The Image

```powershell
docker compose build
```

Builds the local image named `metrogenn:local` using `Dockerfile`.
This installs dependencies and downloads the default SentenceTransformer model
into the image cache.

Use this when Docker files or `requirements.txt` change.

```powershell
docker compose build --no-cache
```

Forces a clean rebuild without using cached Docker layers. This is slower, but
useful when you want to prove the image can be rebuilt from scratch.

## First Run On A New Machine

```powershell
docker compose build
docker compose run --rm metrogenn
```

The default run first validates the configured CSV, graph/model/topic artifacts,
and Chroma vector store. If `models` is empty on a new machine, it bootstraps the
runtime artifacts with seed `42`, builds the vector store, and then runs the demo
inference query.

This requires the bundled CSV at:

```powershell
train/DBLP-Citation-network-V18-1-20-25_clean.csv
```

You can run the bootstrap step explicitly with:

```powershell
docker compose run --rm metrogenn python scripts/bootstrap_artifacts.py
```

## Run Default Inference

```powershell
docker compose run --rm metrogenn
```

Runs the default command from `docker-compose.yml`, which validates or creates
runtime artifacts and then executes:

```powershell
python scripts/run_inference.py
```

Use this as the quick smoke test that the image, mounted data, mounted models,
vector store, and inference path work.

## Run A Custom Query

```powershell
docker compose run --rm metrogenn python -m gnn_pipeline.inference -q "graph neural networks" -t 3 -k 10 --output /tmp/inference.json
```

Runs the inference CLI directly.

- `-q` is the research query.
- `-t` is the number of final papers to show.
- `-k` is the number of vector-store retrieval candidates.
- `--output` writes JSON output inside the container.

Because `/tmp/inference.json` is inside the temporary container, that file is
removed when `--rm` deletes the container. To keep the JSON output on your host
machine, write it to the mounted `saved_results` folder instead:

```powershell
docker compose run --rm metrogenn python -m gnn_pipeline.inference -q "graph neural networks" -t 3 -k 10 --output /app/saved_results/inference.json
```

This creates `saved_results/inference.json` in the project folder on your
machine. The container path `/app/saved_results` is mapped to the local
`./saved_results` folder by `docker-compose.yml`.

## Run The Lightweight Validation Suite

```powershell
docker compose run --rm metrogenn python scripts/test_pipeline.py
```

Runs the non-mutating validation suite:

- unit tests for parsing, topic artifacts, graph/checkpoint contracts, temporal
  recommendation metrics, statistics, vector-store manifests, and MCP schemas
- no embedding generation
- no model training
- no artifact overwrite

Use `scripts/train_pipeline.py` explicitly when you want to rebuild artifacts.

## Train The Model

```powershell
docker compose run --rm metrogenn python scripts/train_pipeline.py
```

Runs the training pipeline using values from `config.yaml`.
This is heavier than `scripts/test_pipeline.py` because it uses the configured
training settings, including the configured epoch count.
Training is seeded by default with `--seed 42`; pass a different `--seed` only
when you intentionally want a new checkpoint.

Generated artifacts are written to the mounted `models` folder.

## Evaluate The Model

```powershell
docker compose run --rm metrogenn python scripts/evaluate_model.py --device cpu --save-dir /tmp/metrogenn_eval
```

Runs model evaluation on CPU and writes reports inside the container at
`/tmp/metrogenn_eval`.

To write evaluation outputs to the project `saved_results` folder instead, omit
`--save-dir`:

```powershell
docker compose run --rm metrogenn python scripts/evaluate_model.py --device cpu
```

## Evaluate Recommendation Baselines

```powershell
docker compose run --rm metrogenn python scripts/build_recommendation_benchmark.py
```

Builds the temporal citation benchmark configured by
`configs/experiments/citation_temporal_v1.yaml`. With the bundled legacy
metadata-only CSV, this writes a clearly labeled citation-proxy pilot benchmark
for end-to-end smoke testing. Use a CSV with real reference lists for paper
evidence.

```powershell
docker compose run --rm metrogenn python scripts/evaluate_recommenders.py --proposed seeded_ppr --baseline tfidf
```

Runs the configured baseline methods, writes per-query metrics and confidence
intervals, and emits paper-ready report/table files under `saved_results`.

## Evaluate MCP Agent Traces

```powershell
docker compose run --rm metrogenn python scripts/evaluate_mcp_agent_traces.py --traces /app/saved_results/agent_traces.jsonl
```

Scores a complete saved trace file against
`agents/benchmarks/mcp_agent_tasks.jsonl`. Use `--allow-partial` only for
debugging incomplete trace captures.

## Profile Training And Serving

```powershell
docker compose run --rm metrogenn python scripts/profile_training.py --steps 20
```

Profiles target construction and warm training-step latency without overwriting
checkpoints.

```powershell
docker compose run --rm metrogenn python scripts/profile_serving.py --runs 20 --query "graph neural networks"
```

Profiles artifact validation/loading, warm GNN forward latency, and optional
end-to-end query latency.

## Health Checks Used During Validation

```powershell
docker compose run --rm metrogenn python -m compileall -q gnn_pipeline scripts mcp_server
```

Compiles Python files inside the container and catches syntax errors.

```powershell
docker compose run --rm metrogenn python -c "import torch, torch_geometric, pandas, chromadb, sentence_transformers, yaml; print('imports ok')"
```

Checks that the important runtime dependencies import correctly.

```powershell
docker compose run --rm metrogenn python -c "from mcp_server.server import mcp; print('mcp import ok', mcp.name)"
```

Checks that the MCP server module imports correctly.

## Open A Shell In The Container

```powershell
docker compose run --rm metrogenn /bin/sh
```

Starts an interactive shell in the container for debugging.

## Check Or Clean Compose State

```powershell
docker compose ps -a
```

Shows containers created by this Compose project.

```powershell
docker compose down
```

Stops and removes Compose containers and the default Compose network. The
project data in `train`, `models`, and `saved_results` stays on the host.

## Mounted Files And Folders

`docker-compose.yml` mounts these host paths:

- `./:/app:ro`
- `./models:/app/models`
- `./saved_results:/app/saved_results`

The project source, `train` folder, and `config.yaml` are read-only inside the
container. The `models` and `saved_results` folders are mounted back over that
read-only project mount as writable paths so training and evaluation outputs
persist on the host.

## Notes

- The image is CPU-only because it installs `torch==2.12.1+cpu`.
- Hugging Face may print an unauthenticated request warning. The model still
  runs, but setting `HF_TOKEN` can improve rate limits.
- Prefer direct command overrides like
  `docker compose run --rm metrogenn python scripts/test_pipeline.py`.
  Passing `-e METROGNN_COMMAND=...` to `docker compose run` does not override the
  Compose command reliably because `${METROGNN_COMMAND}` is expanded by Compose
  before container environment variables are applied.
