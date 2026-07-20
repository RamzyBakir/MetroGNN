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

## Run Default Inference

```powershell
docker compose run --rm metrogenn
```

Runs the default command from `docker-compose.yml`, which executes:

```powershell
python scripts/run_inference.py
```

Use this as the quick smoke test that the image, mounted models, vector store,
and inference path work.

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

## Run The Full Pipeline Test

```powershell
docker compose run --rm metrogenn python scripts/test_pipeline.py
```

Runs the end-to-end integration test:

- loads the DBLP CSV from `train`
- assigns dynamic topics
- creates feature matrices
- builds the knowledge graph
- trains the GNN for a short validation run
- tests dynamic topic detection

This command writes model artifacts into the mounted `models` folder.

## Train The Model

```powershell
docker compose run --rm metrogenn python scripts/train_pipeline.py
```

Runs the training pipeline using values from `config.yaml`.
This is heavier than `scripts/test_pipeline.py` because it uses the configured
training settings, including the configured epoch count.

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

- `./config.yaml:/app/config.yaml:ro`
- `./train:/app/train:ro`
- `./models:/app/models`
- `./saved_results:/app/saved_results`

The `train` folder and `config.yaml` are read-only inside the container.
The `models` and `saved_results` folders are writable so training and evaluation
outputs persist on the host.

## Notes

- The image is CPU-only because it installs `torch==2.12.1+cpu`.
- Hugging Face may print an unauthenticated request warning. The model still
  runs, but setting `HF_TOKEN` can improve rate limits.
- Prefer direct command overrides like
  `docker compose run --rm metrogenn python scripts/test_pipeline.py`.
  Passing `-e METROGNN_COMMAND=...` to `docker compose run` does not override the
  Compose command reliably because `${METROGNN_COMMAND}` is expanded by Compose
  before container environment variables are applied.
