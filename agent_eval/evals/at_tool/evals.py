"""Layer 2 — At/After Tool Call evals (execution quality)."""

import json
from agent_eval.core.types import Trace, Label, EvalResult
from agent_eval.core.registry import register
from agent_eval.core.llm import chat_json
from agent_eval.agent.tools import TOOL_SCHEMAS

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


@register(name="param_schema_valid", layer="at_tool", metric="param_schema_valid")
def eval_param_schema_valid(trace: Trace, label: Label) -> EvalResult:
    """Validate that tool parameters conform to the tool's JSON Schema.

    Deterministic schema validation.
    """
    tool_steps = [s for s in trace.steps if s.selected_tool and s.tool_params is not None]
    if not tool_steps:
        return EvalResult(
            trace_id=trace.trace_id, layer="at_tool",
            metric="param_schema_valid", score=0.0,
            explanation="No tool calls with parameters found."
        )

    if not HAS_JSONSCHEMA:
        return EvalResult(
            trace_id=trace.trace_id, layer="at_tool",
            metric="param_schema_valid", score=0.5,
            explanation="jsonschema library not installed."
        )

    results = []
    for step in tool_steps:
        schema = TOOL_SCHEMAS.get(step.selected_tool, {})
        if not schema:
            results.append((False, f"No schema for tool '{step.selected_tool}'"))
            continue
        try:
            jsonschema.validate(instance=step.tool_params, schema=schema)
            results.append((True, ""))
        except jsonschema.ValidationError as e:
            results.append((False, str(e)))

    valid_count = sum(1 for ok, _ in results if ok)
    score = valid_count / len(results) if results else 0.0
    explanations = [msg for ok, msg in results if not ok]

    return EvalResult(
        trace_id=trace.trace_id, layer="at_tool",
        metric="param_schema_valid", score=score,
        explanation=f"{valid_count}/{len(results)} tool calls schema-valid."
            + (" Errors: " + "; ".join(explanations) if explanations else ""),
        details={"per_step": [{"tool": s.selected_tool, "valid": ok, "error": msg} for (ok, msg), s in zip(results, tool_steps)]},
    )


@register(name="param_correctness", layer="at_tool", metric="param_correctness")
def eval_param_correctness(trace: Trace, label: Label) -> EvalResult:
    """LLM-as-judge: are tool parameter values semantically correct for the intent?"""
    tool_steps = [s for s in trace.steps if s.selected_tool and s.tool_params is not None]
    if not tool_steps:
        return EvalResult(
            trace_id=trace.trace_id, layer="at_tool",
            metric="param_correctness", score=0.0,
            explanation="No tool calls with parameters found."
        )

    first = tool_steps[0]
    expected_params = label.expected_params or {}

    messages = [
        {
            "role": "system",
            "content": (
                "You evaluate whether an AI agent passed correct parameters to a tool call. "
                "Compare the actual parameters with the expected parameters. "
                "Score from 0.0 to 1.0 based on semantic correctness "
                "(exact string match not required, but meaning should match). "
                "Respond with JSON: {\"score\": <float>, \"explanation\": \"<brief>\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {trace.input}\n"
                f"Tool: {first.selected_tool}\n"
                f"Actual params: {json.dumps(first.tool_params)}\n"
                f"Expected params: {json.dumps(expected_params)}"
            ),
        },
    ]

    try:
        result = chat_json(messages, temperature=0.0)
        score = float(result.get("score", 0.5))
        explanation = result.get("explanation", "LLM judge evaluation.")
        score = max(0.0, min(1.0, score))
    except Exception as e:
        score = 0.5
        explanation = f"LLM judge evaluation failed: {e}"

    return EvalResult(
        trace_id=trace.trace_id, layer="at_tool",
        metric="param_correctness", score=score,
        explanation=explanation,
        details={"actual": first.tool_params, "expected": expected_params},
    )


@register(name="result_interpretation", layer="at_tool", metric="result_interpretation")
def eval_result_interpretation(trace: Trace, label: Label) -> EvalResult:
    """LLM-as-judge: did the agent correctly interpret the tool result and plan next steps?"""
    tool_steps = [s for s in trace.steps if s.tool_result]
    if not tool_steps:
        return EvalResult(
            trace_id=trace.trace_id, layer="at_tool",
            metric="result_interpretation", score=0.0,
            explanation="No tool result steps found."
        )

    last_tool_step = tool_steps[-1]
    next_step = trace.steps[last_tool_step.step_number] if last_tool_step.step_number < len(trace.steps) else None

    messages = [
        {
            "role": "system",
            "content": (
                "You evaluate whether an AI agent correctly interpreted a tool result "
                "and chose an appropriate next action. Score from 0.0 to 1.0. "
                "Respond with JSON: {\"score\": <float>, \"explanation\": \"<brief>\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {trace.input}\n"
                f"Tool: {last_tool_step.selected_tool}\n"
                f"Tool result: {last_tool_step.tool_result}\n"
                f"Agent's next action: {last_tool_step.next_action if last_tool_step else 'none'}\n"
                f"Next step reasoning: {next_step.reasoning if next_step else 'none'}"
            ),
        },
    ]

    try:
        result = chat_json(messages, temperature=0.0)
        score = float(result.get("score", 0.5))
        explanation = result.get("explanation", "LLM judge evaluation.")
        score = max(0.0, min(1.0, score))
    except Exception as e:
        score = 0.5
        explanation = f"LLM judge evaluation failed: {e}"

    return EvalResult(
        trace_id=trace.trace_id, layer="at_tool",
        metric="result_interpretation", score=score,
        explanation=explanation,
    )
