"""Auditable one-time final-test registration for frozen recommendation models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict

from gnn_pipeline.artifacts import sha256_file, write_json_atomic


FINAL_TEST_REGISTRY_SCHEMA_VERSION = 1


class FinalTestAlreadyRun(RuntimeError):
    pass


class FinalTestRegistry:
    def __init__(self, registry_root: str):
        self.root = Path(registry_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_plan(
        self,
        *,
        frozen_selection_path: str,
        frozen_baseline_manifest_path: str,
        expected_seed_count: int = 5,
        final_context: Dict | None = None,
    ) -> Dict:
        selection_path = Path(frozen_selection_path)
        baseline_path = Path(frozen_baseline_manifest_path)
        selection = json.loads(selection_path.read_text())
        baseline = json.loads(baseline_path.read_text())
        if selection.get("selection_split") != "validation":
            raise ValueError("Final model selection was not produced exclusively on validation")
        seeds = selection.get("repeated_seeds", [])
        if len(seeds) != expected_seed_count or len(set(seeds)) != expected_seed_count:
            raise ValueError(
                f"Final evaluation requires exactly {expected_seed_count} distinct frozen seeds"
            )
        for artifact in selection.get("winner_artifacts", []):
            path = Path(artifact["artifact_path"])
            if not path.is_file() or sha256_file(str(path)) != artifact["artifact_sha256"]:
                raise ValueError("A frozen winning checkpoint is missing or has changed")

        identity = {
            "registry_schema_version": FINAL_TEST_REGISTRY_SCHEMA_VERSION,
            "selection_id": selection["selection_id"],
            "selection_sha256": sha256_file(str(selection_path)),
            "benchmark_id": selection["benchmark_id"],
            "seeds": seeds,
            "baseline_run_id": baseline["run_id"],
            "baseline_manifest_sha256": sha256_file(str(baseline_path)),
            "test_split": "test",
            "final_context": final_context or {},
        }
        experiment_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        plan = {
            **identity,
            "experiment_id": experiment_id,
            "frozen_selection_path": os.path.abspath(selection_path),
            "frozen_baseline_manifest_path": os.path.abspath(baseline_path),
        }
        experiment_dir = self.root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        plan_path = experiment_dir / "final_test_plan.json"
        if plan_path.exists():
            existing = json.loads(plan_path.read_text())
            if existing != plan:
                raise ValueError("Existing final-test plan differs from frozen inputs")
        else:
            write_json_atomic(str(plan_path), plan)
        return plan

    def run_once(
        self,
        plan: Dict,
        callback: Callable[[Dict, str], Dict[str, str]],
    ) -> Dict:
        """Run callback once; callback must return named output artifact paths."""
        experiment_id = plan["experiment_id"]
        directory = self.root / experiment_id
        complete_path = directory / "FINAL_TEST_COMPLETE.json"
        lock_path = directory / "FINAL_TEST_IN_PROGRESS.lock"
        if complete_path.exists():
            raise FinalTestAlreadyRun(
                f"Final test already completed for experiment {experiment_id}"
            )
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise FinalTestAlreadyRun(
                "A final test is already in progress or requires manual audit after failure"
            ) from exc
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(plan, sort_keys=True))

        outputs = callback(dict(plan), str(directory))
        if not outputs:
            raise RuntimeError("Final-test callback returned no persisted artifacts")
        checksums = {}
        for name, output_path in outputs.items():
            path = Path(output_path)
            if not path.is_file():
                raise RuntimeError(f"Final-test output is missing: {name}")
            checksums[name] = {
                "path": os.path.relpath(path, directory),
                "sha256": sha256_file(str(path)),
            }
        completion = {
            **plan,
            "status": "complete",
            "outputs": checksums,
        }
        write_json_atomic(str(complete_path), completion)
        lock_path.unlink()
        return completion
