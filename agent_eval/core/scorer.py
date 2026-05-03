"""Score aggregation across eval layers."""

from agent_eval.core.types import EvalResult


LAYERS = ["pre_tool", "at_tool", "end_to_end"]


def aggregate_scores(
    results: list[EvalResult],
    layer_weights: dict | None = None,
) -> dict:
    """Aggregate eval results into per-layer and overall scores.

    Returns a dict with keys: overall, pre_tool, at_tool, end_to_end, per_metric.
    """
    if layer_weights is None:
        layer_weights = {layer: 1.0 / 3 for layer in LAYERS}

    by_layer: dict[str, list[EvalResult]] = {layer: [] for layer in LAYERS}
    by_metric: dict[str, list[float]] = {}

    for r in results:
        if r.layer in by_layer:
            by_layer[r.layer].append(r)
        by_metric.setdefault(r.metric, []).append(r.score)

    layer_scores = {}
    for layer in LAYERS:
        scores = [r.score for r in by_layer[layer]]
        layer_scores[layer] = sum(scores) / len(scores) if scores else 0.0

    overall = sum(
        layer_scores[layer] * layer_weights.get(layer, 0)
        for layer in LAYERS
    )

    metric_scores = {
        metric: sum(vals) / len(vals) for metric, vals in by_metric.items()
    }

    return {
        "overall": overall,
        "pre_tool": layer_scores["pre_tool"],
        "at_tool": layer_scores["at_tool"],
        "end_to_end": layer_scores["end_to_end"],
        "per_metric": metric_scores,
    }
