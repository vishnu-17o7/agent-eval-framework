"""Trace collector — intercepts agent steps into structured traces."""

import uuid
from datetime import datetime
from agent_eval.core.types import Trace, TraceStep


class TraceCollector:
    """Collects structured traces during agent execution."""

    def __init__(self):
        self.current_trace: Trace | None = None

    def start_run(self, agent_version: str, query: str) -> str:
        """Begin a new trace. Returns the trace_id."""
        run_id = str(uuid.uuid4())[:8]
        self.current_trace = Trace(
            run_id=run_id,
            agent_version=agent_version,
            scenario_id="",
            input=query,
        )
        return self.current_trace.trace_id

    def add_step(
        self,
        step_number: int,
        reasoning: str,
        intent: str,
        selected_tool: str | None,
        tool_params: dict | None,
        tool_result: str | None,
        next_action: str,
    ):
        """Add a step to the current trace."""
        if self.current_trace is None:
            return
        step = TraceStep(
            step_number=step_number,
            reasoning=reasoning,
            intent=intent,
            selected_tool=selected_tool,
            tool_params=tool_params,
            tool_result=tool_result,
            next_action=next_action,
        )
        self.current_trace.steps.append(step)

    def finish_run(self, final_output: str):
        """Mark the trace as complete."""
        if self.current_trace is None:
            return
        self.current_trace.final_output = final_output
        self.current_trace.total_steps = len(self.current_trace.steps)
        self.current_trace.timestamp = datetime.now().isoformat()

    def get_trace(self) -> Trace | None:
        """Get the completed trace."""
        return self.current_trace

    def reset(self):
        """Reset for a new run."""
        self.current_trace = None
