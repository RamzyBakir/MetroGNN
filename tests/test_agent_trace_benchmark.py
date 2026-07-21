import unittest

from gnn_pipeline.evaluation.agent import (
    AgentTaskSpec,
    AgentTrace,
    evaluate_trace,
    summarize_evaluations,
    validate_trace_coverage,
)


class AgentTraceBenchmarkTests(unittest.TestCase):
    def test_trace_scores_tool_sequence_arguments_and_evidence(self):
        task = AgentTaskSpec(
            task_id="t1",
            user_prompt="Search papers",
            required_tools=["query_papers", "run_inference"],
            allowed_tools=["query_papers", "run_inference"],
            required_arguments={
                "query_papers": {"query": "graph neural networks"},
                "run_inference": {"query": "graph neural networks", "top_papers": 3},
            },
            expected_evidence=["Paper A", "Author 7"],
        )
        trace = AgentTrace(
            task_id="t1",
            tool_calls=[
                {
                    "name": "query_papers",
                    "arguments": {"query": "graph neural networks", "top_k": 20},
                    "ok": True,
                    "output": {"papers": [{"title": "Paper A"}]},
                },
                {
                    "name": "run_inference",
                    "arguments": {"query": "graph neural networks", "top_papers": 3},
                    "ok": True,
                    "output": {"summary": "Author 7 is influential"},
                },
            ],
            final_response="Paper A is recommended with Author 7.",
        )
        result = evaluate_trace(task, trace)
        self.assertTrue(result.valid_tool_sequence)
        self.assertEqual(result.parameter_accuracy, 1.0)
        self.assertEqual(result.grounding_score, 1.0)
        self.assertEqual(result.overall_score, 1.0)

    def test_trace_surfaces_missing_tools_and_argument_mismatch(self):
        task = AgentTaskSpec(
            task_id="t2",
            user_prompt="Analyze community",
            required_tools=["detect_topics", "analyze_community"],
            allowed_tools=["detect_topics", "analyze_community"],
            required_arguments={"detect_topics": {"query": "retrieval"}},
            expected_evidence=["community"],
        )
        trace = AgentTrace(
            task_id="t2",
            tool_calls=[
                {
                    "name": "detect_topics",
                    "arguments": {"query": "wrong"},
                    "ok": False,
                    "error": "temporary failure",
                }
            ],
            final_response="No result.",
        )
        result = evaluate_trace(task, trace)
        self.assertIn("analyze_community", result.missing_tools)
        self.assertFalse(result.valid_tool_sequence)
        self.assertEqual(result.parameter_accuracy, 0.0)
        self.assertEqual(result.recovery_score, 0.0)
        self.assertIn("community", result.missing_evidence)

        summary = summarize_evaluations([result])
        self.assertEqual(summary["tasks_evaluated"], 1)
        self.assertLess(summary["overall_score"], 1.0)

    def test_trace_coverage_requires_one_trace_per_task(self):
        tasks = [
            AgentTaskSpec(task_id="a", user_prompt="A"),
            AgentTaskSpec(task_id="b", user_prompt="B"),
        ]
        traces = [AgentTrace(task_id="a", tool_calls=[])]
        with self.assertRaisesRegex(ValueError, "missing traces"):
            validate_trace_coverage(tasks, traces)

        duplicate = [
            AgentTrace(task_id="a", tool_calls=[]),
            AgentTrace(task_id="a", tool_calls=[]),
            AgentTrace(task_id="b", tool_calls=[]),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate traces"):
            validate_trace_coverage(tasks, duplicate)

        validate_trace_coverage(
            tasks,
            [
                AgentTrace(task_id="a", tool_calls=[]),
                AgentTrace(task_id="b", tool_calls=[]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
