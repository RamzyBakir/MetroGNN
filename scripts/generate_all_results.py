"""Orchestrate regeneration of the MetroGNN paper result artifacts.

Runs the result-generation scripts in sequence:

    1. evaluate_model.py            -> model evaluation figures + LaTeX table
    2. visualize_community.py       -> community collaboration network + quality radar
    3. evaluate_recommenders.py     -> recommendation CIs, paired comparisons, report
    4. generate_significance_report.py -> significance figures + markdown/LaTeX report

Assumes the trained model (``models/``) and the recommendation benchmark
(``saved_results/benchmarks/citation_temporal_v1/``) already exist. Build them
first if needed:

    python scripts/train_pipeline.py
    python scripts/build_recommendation_benchmark.py

Every step always runs (no skipping). Output is streamed live and the run stops
at the first failing step.

Outputs (under saved_results/):
    eval_report.txt, confusion_matrices.png, regression_analysis.png,
    tsne_importance.png, tsne_role.png, results_table.tex
    community_network_<query>.png, community_quality_<query>.png
    evaluations/citation_temporal_v1/<id>/{report.md,results_table.tex,...}
    confidence_intervals.png, significance_tests.png,
    statistical_report.md, significance_report.tex

Usage:
    python scripts/generate_all_results.py
    python scripts/generate_all_results.py --device cpu --query "graph neural networks"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_step(step: str, cmd: list) -> None:
    print("\n" + "=" * 70)
    print(f"=== STEP: {step} ===")
    print("=" * 70)
    print("  $ " + " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Step '{step}' failed (exit={proc.returncode}).")
    print(f"  [ok] {step} ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the model-based steps (default: cpu)",
    )
    parser.add_argument(
        "--query",
        default="AI agents",
        help="Query used for the community visualization (default: 'AI agents')",
    )
    parser.add_argument(
        "--proposed",
        default="seeded_ppr",
        help="Proposed method for the paired recommendation comparison",
    )
    parser.add_argument(
        "--baseline",
        default="tfidf",
        help="Baseline method for the paired recommendation comparison",
    )
    args = parser.parse_args()

    steps = [
        (
            "evaluate_model",
            [PY, "scripts/evaluate_model.py", "--device", args.device],
        ),
        (
            "visualize_community",
            [
                PY,
                "scripts/visualize_community.py",
                "--query",
                args.query,
                "--device",
                args.device,
            ],
        ),
        (
            "evaluate_recommenders",
            [
                PY,
                "scripts/evaluate_recommenders.py",
                "--proposed",
                args.proposed,
                "--baseline",
                args.baseline,
            ],
        ),
        (
            "generate_significance_report",
            [PY, "scripts/generate_significance_report.py"],
        ),
    ]

    total_start = time.time()
    for step, cmd in steps:
        run_step(step, cmd)

    print("\n" + "=" * 70)
    print(f"ALL RESULTS GENERATED in {time.time() - total_start:.1f}s")
    print("=" * 70)
    print("Outputs are under saved_results/ (see each step's output above).")


if __name__ == "__main__":
    main()
