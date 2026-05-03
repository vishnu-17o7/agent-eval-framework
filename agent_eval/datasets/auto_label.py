"""Auto-generate labels for evaluation scenarios.

This script creates reasonable default labels for all 12 scenarios
so users don't have to manually label through the interactive CLI.
"""

import json
from pathlib import Path
from agent_eval.core.types import Label
from agent_eval.datasets.manager import DatasetManager

LABEL_MAP = {
    "s01": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "climate change"},
        goal_achieved=True,
        optimal_steps=2,
    ),
    "s02": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "Python programming language"},
        goal_achieved=True,
        optimal_steps=2,
    ),
    "s03": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "electric cars"},
        goal_achieved=True,
        optimal_steps=2,
    ),
    "s04": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "machine learning"},
        goal_achieved=True,
        optimal_steps=2,
    ),
    "s05": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "space exploration"},
        goal_achieved=True,
        optimal_steps=2,
    ),
    "s06": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "renewable energy"},
        goal_achieved=True,
        optimal_steps=3,
    ),
    "s07": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "quantum computing"},
        goal_achieved=True,
        optimal_steps=2,
    ),
    "s08": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "artificial intelligence"},
        goal_achieved=True,
        optimal_steps=3,
    ),
    "s09": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "climate change"},
        goal_achieved=True,
        optimal_steps=3,
    ),
    "s10": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "electric cars"},
        goal_achieved=True,
        optimal_steps=2,
    ),
    "s11": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "Python"},
        goal_achieved=True,
        optimal_steps=4,
    ),
    "s12": Label(
        expected_intent="search_information",
        expected_tool="search",
        expected_params={"query": "space exploration"},
        goal_achieved=True,
        optimal_steps=3,
    ),
}


def auto_label(version: str):
    """Generate and save labels for a dataset version, mapping labels to trace IDs."""
    mgr = DatasetManager()
    traces = mgr.load_traces(version)

    if not traces:
        print(f"No traces found for version '{version}'.")
        return

    labels = {}
    for trace in traces:
        scenario_id = trace.scenario_id
        if scenario_id in LABEL_MAP:
            base_label = LABEL_MAP[scenario_id]
            label = Label(
                trace_id=trace.trace_id,
                expected_intent=base_label.expected_intent,
                expected_tool=base_label.expected_tool,
                expected_params=base_label.expected_params,
                goal_achieved=base_label.goal_achieved,
                optimal_steps=base_label.optimal_steps,
            )
            labels[trace.trace_id] = label

    mgr.save_labels(labels, version)
    print(f"Auto-generated {len(labels)} labels for version '{version}'.")
    print(f"Scenarios labeled: {list(labels.keys())}")


if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    auto_label(version)
