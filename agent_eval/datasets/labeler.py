"""Interactive labeling CLI for evaluation datasets."""

import json
import sys
from agent_eval.core.types import Trace, Label
from agent_eval.datasets.manager import DatasetManager


def label_traces(version: str = "v1"):
    """Interactive CLI to label traces with expected intent, tool, params, etc.

    Walks through each trace and prompts the user to provide ground-truth labels.
    """
    mgr = DatasetManager()
    traces = mgr.load_traces(version)
    if not traces:
        print(f"No traces found for version '{version}'. Run the agent first.")
        return

    existing_labels = mgr.load_labels(version)
    labels: dict[str, Label] = {}

    print(f"\nLabeling {len(traces)} traces for version '{version}'")
    print("Press Enter to accept defaults, or type 'skip' to skip a trace.\n")

    for i, trace in enumerate(traces):
        print(f"[{i+1}/{len(traces)}] Trace: {trace.trace_id[:8]}...")
        print(f"  Query: {trace.input}")
        print(f"  Output: {trace.final_output[:200]}...")
        print(f"  Steps: {trace.total_steps}")
        if trace.steps:
            s = trace.steps[0]
            print(f"  First step intent: {s.intent}")
            print(f"  First tool: {s.selected_tool}")
            print(f"  First params: {json.dumps(s.tool_params) if s.tool_params else 'none'}")

        existing = existing_labels.get(trace.trace_id)
        default_intent = existing.expected_intent if existing else (trace.steps[0].intent if trace.steps else "")
        default_tool = existing.expected_tool if existing else (trace.steps[0].selected_tool or "")
        default_params = existing.expected_params if existing else (trace.steps[0].tool_params or {})
        default_goal = "y" if (existing and existing.goal_achieved) else "n"
        default_steps = str(existing.optimal_steps if existing else 2)

        intent = input(f"  Expected intent [{default_intent}]: ").strip()
        if intent.lower() == "skip":
            continue
        intent = intent or default_intent

        tool = input(f"  Expected tool [{default_tool}]: ").strip() or default_tool

        params_str = input(f"  Expected params (JSON) [{json.dumps(default_params)}]: ").strip()
        try:
            params = json.loads(params_str) if params_str else default_params
        except json.JSONDecodeError:
            params = default_params

        goal = input(f"  Goal achieved? (y/n) [{default_goal}]: ").strip().lower() or default_goal
        goal_achieved = goal == "y"

        steps_str = input(f"  Optimal steps [{default_steps}]: ").strip() or default_steps
        optimal_steps = int(steps_str) if steps_str.isdigit() else 2

        label = Label(
            trace_id=trace.trace_id,
            expected_intent=intent,
            expected_tool=tool,
            expected_params=params,
            goal_achieved=goal_achieved,
            optimal_steps=optimal_steps,
        )
        labels[trace.trace_id] = label
        print(f"  Saved: intent={intent}, tool={tool}, goal={goal_achieved}, steps={optimal_steps}\n")

    mgr.save_labels(labels, version)
    print(f"Saved {len(labels)} labels to dataset version '{version}'.")
