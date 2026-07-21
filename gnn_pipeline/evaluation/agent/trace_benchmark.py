"""Evaluate fixed MCP/agent task traces.

The project does not yet implement a live LLM agent. This module provides the
reproducible harness needed to evaluate one later from saved tool-call traces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


AGENT_BENCHMARK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentTaskSpec:
    task_id: str
    user_prompt: str
    required_tools: List[str] = field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    required_arguments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    expected_evidence: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentTaskSpec":
        if not value.get("task_id"):
            raise ValueError("Agent task spec is missing task_id")
        return cls(
            task_id=str(value["task_id"]),
            user_prompt=str(value.get("user_prompt", "")),
            required_tools=[str(tool) for tool in value.get("required_tools", [])],
            allowed_tools=(
                [str(tool) for tool in value["allowed_tools"]]
                if value.get("allowed_tools") is not None
                else None
            ),
            required_arguments={
                str(tool): dict(args)
                for tool, args in value.get("required_arguments", {}).items()
            },
            expected_evidence=[
                str(item) for item in value.get("expected_evidence", [])
            ],
        )


@dataclass(frozen=True)
class AgentTrace:
    task_id: str
    tool_calls: List[Dict[str, Any]]
    final_response: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentTrace":
        if not value.get("task_id"):
            raise ValueError("Agent trace is missing task_id")
        calls = value.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("Agent trace tool_calls must be a list")
        return cls(
            task_id=str(value["task_id"]),
            tool_calls=[dict(call) for call in calls],
            final_response=str(value.get("final_response", "")),
        )


@dataclass(frozen=True)
class TraceEvaluation:
    task_id: str
    tool_selection_score: float
    parameter_accuracy: float
    valid_tool_sequence: bool
    recovery_score: float
    grounding_score: float
    overall_score: float
    missing_tools: List[str] = field(default_factory=list)
    unexpected_tools: List[str] = field(default_factory=list)
    parameter_errors: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_selection_score": self.tool_selection_score,
            "parameter_accuracy": self.parameter_accuracy,
            "valid_tool_sequence": self.valid_tool_sequence,
            "recovery_score": self.recovery_score,
            "grounding_score": self.grounding_score,
            "overall_score": self.overall_score,
            "missing_tools": self.missing_tools,
            "unexpected_tools": self.unexpected_tools,
            "parameter_errors": self.parameter_errors,
            "missing_evidence": self.missing_evidence,
        }


def _tool_name(call: Mapping[str, Any]) -> str:
    return str(call.get("name") or call.get("tool") or call.get("tool_name") or "")


def _tool_arguments(call: Mapping[str, Any]) -> Dict[str, Any]:
    args = call.get("arguments", call.get("args", {}))
    return dict(args) if isinstance(args, Mapping) else {}


def _call_succeeded(call: Mapping[str, Any]) -> bool:
    if "ok" in call:
        return bool(call["ok"])
    if "success" in call:
        return bool(call["success"])
    return not bool(call.get("error"))


def _is_subsequence(required: Sequence[str], actual: Sequence[str]) -> bool:
    cursor = 0
    for tool in actual:
        if cursor < len(required) and tool == required[cursor]:
            cursor += 1
    return cursor == len(required)


def _matches_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _matches_subset(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and expected == actual
    return expected == actual


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def evaluate_trace(task: AgentTaskSpec, trace: AgentTrace) -> TraceEvaluation:
    if task.task_id != trace.task_id:
        raise ValueError(f"Trace {trace.task_id!r} does not match task {task.task_id!r}")

    call_names = [_tool_name(call) for call in trace.tool_calls]
    call_name_set = set(call_names)
    required_set = set(task.required_tools)
    missing_tools = [tool for tool in task.required_tools if tool not in call_name_set]
    unexpected_tools: List[str] = []
    if task.allowed_tools is not None:
        allowed = set(task.allowed_tools)
        unexpected_tools = [tool for tool in call_names if tool and tool not in allowed]

    if task.required_tools:
        tool_selection_score = (
            len(required_set - set(missing_tools)) / len(required_set)
        )
    else:
        tool_selection_score = 1.0

    valid_sequence = _is_subsequence(task.required_tools, call_names) and not unexpected_tools

    parameter_errors = []
    checked = 0
    passed = 0
    for tool, expected_args in task.required_arguments.items():
        checked += 1
        matching_call = next((call for call in trace.tool_calls if _tool_name(call) == tool), None)
        if matching_call is None:
            parameter_errors.append(f"{tool}: missing call")
            continue
        actual_args = _tool_arguments(matching_call)
        if _matches_subset(expected_args, actual_args):
            passed += 1
        else:
            parameter_errors.append(f"{tool}: argument mismatch")
    parameter_accuracy = passed / checked if checked else 1.0

    failed_indices = [
        index for index, call in enumerate(trace.tool_calls) if not _call_succeeded(call)
    ]
    if not failed_indices:
        recovery_score = 1.0
    else:
        first_failure = min(failed_indices)
        recovered = any(
            _call_succeeded(call) for call in trace.tool_calls[first_failure + 1 :]
        )
        recovery_score = 1.0 if recovered and trace.final_response.strip() else 0.0

    evidence_haystack = trace.final_response + "\n" + "\n".join(
        _stringify(call.get("output") or call.get("result"))
        for call in trace.tool_calls
        if _call_succeeded(call)
    )
    evidence_haystack = evidence_haystack.casefold()
    missing_evidence = [
        item
        for item in task.expected_evidence
        if item.casefold() not in evidence_haystack
    ]
    grounding_score = (
        (len(task.expected_evidence) - len(missing_evidence))
        / len(task.expected_evidence)
        if task.expected_evidence
        else 1.0
    )

    overall = float(
        sum(
            [
                tool_selection_score,
                parameter_accuracy,
                1.0 if valid_sequence else 0.0,
                recovery_score,
                grounding_score,
            ]
        )
        / 5.0
    )

    return TraceEvaluation(
        task_id=task.task_id,
        tool_selection_score=float(tool_selection_score),
        parameter_accuracy=float(parameter_accuracy),
        valid_tool_sequence=valid_sequence,
        recovery_score=float(recovery_score),
        grounding_score=float(grounding_score),
        overall_score=overall,
        missing_tools=missing_tools,
        unexpected_tools=unexpected_tools,
        parameter_errors=parameter_errors,
        missing_evidence=missing_evidence,
    )


def evaluate_traces(
    tasks: Sequence[AgentTaskSpec], traces: Sequence[AgentTrace]
) -> List[TraceEvaluation]:
    task_by_id = {task.task_id: task for task in tasks}
    evaluations = []
    for trace in traces:
        if trace.task_id not in task_by_id:
            raise KeyError(f"Trace references unknown task_id: {trace.task_id}")
        evaluations.append(evaluate_trace(task_by_id[trace.task_id], trace))
    return evaluations


def validate_trace_coverage(
    tasks: Sequence[AgentTaskSpec], traces: Sequence[AgentTrace]
) -> None:
    """Require exactly one trace for each fixed benchmark task."""
    task_ids = [task.task_id for task in tasks]
    task_id_set = set(task_ids)
    trace_ids = [trace.task_id for trace in traces]
    trace_id_set = set(trace_ids)

    missing = [task_id for task_id in task_ids if task_id not in trace_id_set]
    unknown = sorted(trace_id_set - task_id_set)
    duplicates = sorted(
        task_id for task_id in trace_id_set if trace_ids.count(task_id) > 1
    )
    if missing or unknown or duplicates:
        details = []
        if missing:
            details.append(f"missing traces for {missing}")
        if unknown:
            details.append(f"unknown task IDs {unknown}")
        if duplicates:
            details.append(f"duplicate traces for {duplicates}")
        raise ValueError("Trace file does not cover the benchmark: " + "; ".join(details))


def summarize_evaluations(evaluations: Sequence[TraceEvaluation]) -> Dict[str, Any]:
    if not evaluations:
        return {
            "schema_version": AGENT_BENCHMARK_SCHEMA_VERSION,
            "tasks_evaluated": 0,
            "overall_score": 0.0,
        }
    metrics = [
        "tool_selection_score",
        "parameter_accuracy",
        "recovery_score",
        "grounding_score",
        "overall_score",
    ]
    summary = {
        "schema_version": AGENT_BENCHMARK_SCHEMA_VERSION,
        "tasks_evaluated": len(evaluations),
        "valid_tool_sequence_rate": sum(e.valid_tool_sequence for e in evaluations)
        / len(evaluations),
    }
    for metric in metrics:
        summary[metric] = sum(getattr(evaluation, metric) for evaluation in evaluations) / len(evaluations)
    return summary


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def load_task_specs(path: str) -> List[AgentTaskSpec]:
    return [AgentTaskSpec.from_dict(record) for record in _load_jsonl(path)]


def load_traces(path: str) -> List[AgentTrace]:
    return [AgentTrace.from_dict(record) for record in _load_jsonl(path)]


def evaluations_to_rows(evaluations: Iterable[TraceEvaluation]) -> List[Dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        row = evaluation.to_dict()
        for key in ("missing_tools", "unexpected_tools", "parameter_errors", "missing_evidence"):
            row[key] = json.dumps(row[key], sort_keys=True)
        rows.append(row)
    return rows
