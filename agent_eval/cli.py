"""CLI entry point for the Agent Evaluation Framework.

Commands:
  agent-eval run-scenarios   Run agent on all scenarios and capture traces
  agent-eval label           Interactive labeling of traces
  agent-eval eval            Run evals against labeled datasets
  agent-eval report          Print a text-based score report
  agent-eval dashboard       Launch the Streamlit dashboard
"""

import argparse
import sys
import json
from pathlib import Path

from agent_eval.core.types import Scenario
from agent_eval.agent.agent import ResearchAgent
from agent_eval.tracing.collector import TraceCollector
from agent_eval.tracing.store import save_trace, save_score_history
from agent_eval.datasets.manager import DatasetManager
from agent_eval.datasets.labeler import label_traces
from agent_eval.datasets.auto_label import auto_label
from agent_eval.core.runner import EvalRunner
import agent_eval.evals  # registers all evals


def cmd_run_scenarios(args):
    """Run the agent on all scenarios and capture traces."""
    version = args.version
    mgr = DatasetManager()
    scenarios = mgr.load_scenarios("v1")  # scenarios are version-independent

    if not scenarios:
        print("No scenarios found. Run with --generate to create default scenarios.")
        if args.generate:
            scenarios = _default_scenarios()
            mgr.save_scenarios(scenarios, "v1")
            print(f"Generated {len(scenarios)} default scenarios.")
        else:
            return

    agent = ResearchAgent(version=version)
    collector = TraceCollector()
    traces = []

    print(f"\nRunning agent {version} on {len(scenarios)} scenarios...")
    for i, scenario in enumerate(scenarios):
        print(f"  [{i+1}/{len(scenarios)}] {scenario.scenario_id}: {scenario.query[:60]}...")
        collector.reset()
        try:
            answer = agent.run(scenario.query, collector)
        except Exception as e:
            print(f"    ERROR: {e}")
            collector.finish_run(f"ERROR: {e}")
            answer = f"ERROR: {e}"

        trace = collector.get_trace()
        if trace:
            trace.scenario_id = scenario.scenario_id
            save_trace(trace)
            traces.append(trace)

        print(f"    Steps: {trace.total_steps if trace else 0}, "
              f"Answer: {answer[:80]}...")

    mgr.save_traces(traces, version)
    print(f"\nSaved {len(traces)} traces to dataset version '{version}'.")


def cmd_label(args):
    """Interactive labeling of traces."""
    label_traces(args.version)


def cmd_auto_label(args):
    """Auto-generate labels for all scenarios."""
    auto_label(args.version)


def cmd_eval(args):
    """Run evals against labeled datasets."""
    version = args.version
    mgr = DatasetManager()

    traces = mgr.load_traces(version)
    labels = mgr.load_labels(version)

    if not traces:
        print(f"No traces found for version '{version}'.")
        return
    if not labels:
        print(f"No labels found for version '{version}'. Run 'agent-eval label' first.")
        return

    print(f"\nRunning evals on {len(traces)} traces (version '{version}')...")

    runner = EvalRunner()
    summary = runner.run_and_summarize(traces, labels)

    print("\n" + "=" * 60)
    print(f"  EVAL RESULTS — Agent {version}")
    print("=" * 60)
    print(f"  Overall Score:   {summary['overall']:.3f}")
    print(f"  Pre-Tool:        {summary['pre_tool']:.3f}")
    print(f"  At-Tool:         {summary['at_tool']:.3f}")
    print(f"  End-to-End:      {summary['end_to_end']:.3f}")
    print("-" * 60)
    for metric, score in summary["per_metric"].items():
        print(f"  {metric:25s} {score:.3f}")
    print("=" * 60)

    save_score_history(
        agent_version=version,
        overall=summary["overall"],
        pre_tool=summary["pre_tool"],
        at_tool=summary["at_tool"],
        end_to_end=summary["end_to_end"],
        metric_scores=summary["per_metric"],
    )

    gate_threshold = args.gate or 0.6
    passed = summary["overall"] >= gate_threshold
    print(f"\nGate threshold: {gate_threshold:.2f} — {'PASSED' if passed else 'FAILED'}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output}")


def cmd_report(args):
    """Print a text-based score report."""
    from agent_eval.tracing.store import get_score_history
    history = get_score_history()

    if not history:
        print("No score history found. Run 'agent-eval eval' first.")
        return

    print("\n" + "=" * 70)
    print("  SCORE HISTORY")
    print("=" * 70)
    print(f"  {'Version':<10s} {'Overall':>8s} {'Pre-Tool':>8s} {'At-Tool':>8s} {'E2E':>8s}  Timestamp")
    print("-" * 70)
    for h in history:
        print(
            f"  {h['agent_version']:<10s} "
            f"{h['overall_score']:>8.3f} "
            f"{h['pre_tool_score']:>8.3f} "
            f"{h['at_tool_score']:>8.3f} "
            f"{h['end_to_end_score']:>8.3f}  "
            f"{h['timestamp'][:19]}"
        )
    print("=" * 70)


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port or 8501),
    ])


def _default_scenarios() -> list[Scenario]:
    """Generate 12 default evaluation scenarios."""
    return [
        Scenario("s01", "What is climate change and what causes it?", "Factual knowledge lookup"),
        Scenario("s02", "Tell me about the Python programming language.", "Factual knowledge lookup"),
        Scenario("s03", "Search for information about electric cars.", "Web search task"),
        Scenario("s04", "What is machine learning? Give me a detailed explanation.", "Detailed knowledge task"),
        Scenario("s05", "Look up information about space exploration milestones.", "Factual lookup"),
        Scenario("s06", "Search for renewable energy and summarize what you find.", "Search + summarize"),
        Scenario("s07", "What can you tell me about quantum computing?", "Knowledge lookup"),
        Scenario("s08", "Find information about artificial intelligence and summarize it.", "Search + summarize"),
        Scenario("s09", "Tell me about climate change and then summarize the key points.", "Multi-step task"),
        Scenario("s10", "Search for electric cars and then explain their benefits.", "Search + explain"),
        Scenario("s11", "What is Python? Also tell me about machine learning.", "Multi-topic task"),
        Scenario("s12", "Find information about space exploration and summarize the highlights.", "Search + summarize"),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Agent Evaluation Framework CLI",
        prog="agent-eval",
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    run_parser = sub.add_parser("run-scenarios", help="Run agent on scenarios and capture traces")
    run_parser.add_argument("--version", "-v", default="v1", help="Agent version (v1 or v2)")
    run_parser.add_argument("--generate", "-g", action="store_true", help="Generate default scenarios if none exist")

    label_parser = sub.add_parser("label", help="Interactively label traces")
    label_parser.add_argument("--version", "-v", default="v1", help="Dataset version to label")

    auto_label_parser = sub.add_parser("auto-label", help="Auto-generate labels from predefined map")
    auto_label_parser.add_argument("--version", "-v", default="v1", help="Dataset version to label")

    eval_parser = sub.add_parser("eval", help="Run evals against labeled dataset")
    eval_parser.add_argument("--version", "-v", default="v1", help="Dataset version to evaluate")
    eval_parser.add_argument("--gate", "-g", type=float, default=0.6, help="Gate threshold (default: 0.6)")
    eval_parser.add_argument("--output", "-o", help="Output JSON file for results")

    report_parser = sub.add_parser("report", help="Print score history report")

    dash_parser = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", "-p", type=int, default=8501, help="Dashboard port")

    args = parser.parse_args()

    if args.command == "run-scenarios":
        cmd_run_scenarios(args)
    elif args.command == "label":
        cmd_label(args)
    elif args.command == "auto-label":
        cmd_auto_label(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
