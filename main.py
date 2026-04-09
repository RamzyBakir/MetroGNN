from gnn_pipeline.inference import print_results, run_inference

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
