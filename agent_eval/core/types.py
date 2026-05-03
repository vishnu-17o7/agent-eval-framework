"""Core data types for the Agent Evaluation Framework."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid


@dataclass
class TraceStep:
    step_number: int
    reasoning: str
    intent: str
    selected_tool: Optional[str]
    tool_params: Optional[dict]
    tool_result: Optional[str]
    next_action: str


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    agent_version: str = "v1"
    scenario_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input: str = ""
    final_output: str = ""
    total_steps: int = 0
    steps: list[TraceStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "agent_version": self.agent_version,
            "scenario_id": self.scenario_id,
            "timestamp": self.timestamp,
            "input": self.input,
            "final_output": self.final_output,
            "total_steps": self.total_steps,
            "steps": [
                {
                    "step_number": s.step_number,
                    "reasoning": s.reasoning,
                    "intent": s.intent,
                    "selected_tool": s.selected_tool,
                    "tool_params": s.tool_params,
                    "tool_result": s.tool_result,
                    "next_action": s.next_action,
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        trace = cls(
            trace_id=d["trace_id"],
            run_id=d.get("run_id", ""),
            agent_version=d.get("agent_version", "v1"),
            scenario_id=d.get("scenario_id", ""),
            timestamp=d.get("timestamp", ""),
            input=d.get("input", ""),
            final_output=d.get("final_output", ""),
            total_steps=d.get("total_steps", 0),
        )
        trace.steps = [
            TraceStep(
                step_number=s["step_number"],
                reasoning=s.get("reasoning", ""),
                intent=s.get("intent", ""),
                selected_tool=s.get("selected_tool"),
                tool_params=s.get("tool_params"),
                tool_result=s.get("tool_result"),
                next_action=s.get("next_action", ""),
            )
            for s in d.get("steps", [])
        ]
        return trace


@dataclass
class EvalResult:
    eval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    layer: str = ""  # "pre_tool", "at_tool", "end_to_end"
    metric: str = ""
    score: float = 0.0
    explanation: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "trace_id": self.trace_id,
            "layer": self.layer,
            "metric": self.metric,
            "score": self.score,
            "explanation": self.explanation,
            "details": self.details,
        }


@dataclass
class Label:
    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    expected_intent: str = ""
    expected_tool: str = ""
    expected_params: dict = field(default_factory=dict)
    goal_achieved: bool = False
    optimal_steps: int = 1

    def to_dict(self) -> dict:
        return {
            "label_id": self.label_id,
            "trace_id": self.trace_id,
            "expected_intent": self.expected_intent,
            "expected_tool": self.expected_tool,
            "expected_params": self.expected_params,
            "goal_achieved": self.goal_achieved,
            "optimal_steps": self.optimal_steps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Label":
        return cls(
            label_id=d.get("label_id", ""),
            trace_id=d.get("trace_id", ""),
            expected_intent=d.get("expected_intent", ""),
            expected_tool=d.get("expected_tool", ""),
            expected_params=d.get("expected_params", {}),
            goal_achieved=d.get("goal_achieved", False),
            optimal_steps=d.get("optimal_steps", 1),
        )


@dataclass
class Scenario:
    scenario_id: str
    query: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "query": self.query,
            "description": self.description,
        }
