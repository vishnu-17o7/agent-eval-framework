"""EvalRunner — orchestrates evals against agent runs."""

from agent_eval.core.types import Trace, Label, EvalResult
from agent_eval.core.registry import get_all, get_by_layer
from agent_eval.core.scorer import aggregate_scores


class EvalRunner:
    """Runs all registered evals against a set of traces and labels."""

    def __init__(self, layer_weights: dict | None = None):
        self.layer_weights = layer_weights

    def evaluate_one(self, trace: Trace, label: Label) -> list[EvalResult]:
        """Run all evals against a single trace + label pair."""
        results = []
        for evaluator in get_all():
            result = evaluator.fn(trace, label)
            results.append(result)
        return results

    def evaluate_all(
        self, traces: list[Trace], labels: dict[str, Label]
    ) -> list[EvalResult]:
        """Run all evals across many traces."""
        all_results = []
        for trace in traces:
            label = labels.get(trace.trace_id)
            if label is None:
                continue
            all_results.extend(self.evaluate_one(trace, label))
        return all_results

    def run_and_summarize(
        self, traces: list[Trace], labels: dict[str, Label]
    ) -> dict:
        """Run all evals and return aggregated scores."""
        results = self.evaluate_all(traces, labels)
        return aggregate_scores(results, self.layer_weights)
