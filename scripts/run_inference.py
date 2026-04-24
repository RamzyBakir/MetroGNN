import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_pipeline.config import ensure_dirs, get_path
from gnn_pipeline.inference import print_results, run_inference

ensure_dirs()

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
)

print_results(result)
