import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_pipeline.config import ensure_dirs
from gnn_pipeline.inference import print_results, run_inference
from scripts.bootstrap_artifacts import ensure_runtime_artifacts

ensure_dirs()
device = os.environ.get("METROGNN_DEVICE", "auto")
ensure_runtime_artifacts(device=device)

result = run_inference(
    query="AI agents",
    experience_level="intermediate",
    interests=[
        "AI agents",
        "autonomous agents",
        "multi-agent systems",
        "agentic AI",
        "reinforcement learning",
        "planning and reasoning",
    ],
    goals="research",
    top_papers=10,
    retrieval_k=100,
    sort_by="influence",
    device=device,
)

print_results(result)
