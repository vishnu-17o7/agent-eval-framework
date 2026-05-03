"""Query API for traces."""


def filter_traces_by_layer(traces: list, layer: str) -> list:
    """Filter traces that have steps relevant to a given layer.

    'pre_tool' — traces with at least one tool call step
    'at_tool' — traces with at least one completed tool call
    'end_to_end' — all traces (they all have final output)
    """
    if layer == "end_to_end":
        return traces
    if layer == "pre_tool":
        return [t for t in traces if any(s.selected_tool for s in t.steps)]
    if layer == "at_tool":
        return [t for t in traces if any(s.tool_result for s in t.steps)]
    return traces
