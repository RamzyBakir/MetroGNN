"""Run MetroGNN's lightweight, non-mutating validation suite.

This command intentionally does not generate embeddings, overwrite artifacts, or
train a model. Use ``scripts/train_pipeline.py`` explicitly for those operations.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_tests() -> bool:
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
