"""Layer 3 — End-to-End evals (goal completion, efficiency, hallucination)."""

from agent_eval.core.types import Trace, Label, EvalResult
from agent_eval.core.registry import register
from agent_eval.core.llm import chat_json


@register(name="goal_completion", layer="end_to_end", metric="goal_completion")
def eval_goal_completion(trace: Trace, label: Label) -> EvalResult:
    """LLM-as-judge: did the agent accomplish the user's goal?

    Returns a binary score (0 or 1) with confidence.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You evaluate whether an AI agent successfully completed the user's request. "
                "Compare the agent's final output to the user's query. "
                "Score 1.0 if the goal was achieved, 0.0 if it was not. "
                "Be strict: the answer must address the core of the question. "
                "Respond with JSON: {\"score\": <0.0 or 1.0>, \"explanation\": \"<brief>\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {trace.input}\n\nAgent final output:\n{trace.final_output}"
            ),
        },
    ]

    try:
        result = chat_json(messages, temperature=0.0)
        score = float(result.get("score", 0.0))
        explanation = result.get("explanation", "LLM judge evaluation.")
        score = 1.0 if score >= 0.5 else 0.0
    except Exception as e:
        score = 0.0
        explanation = f"LLM judge evaluation failed: {e}"

    return EvalResult(
        trace_id=trace.trace_id, layer="end_to_end",
        metric="goal_completion", score=score,
        explanation=explanation,
    )


@register(name="efficiency_score", layer="end_to_end", metric="efficiency_score")
def eval_efficiency(trace: Trace, label: Label) -> EvalResult:
    """Deterministic: ratio of optimal steps to actual steps taken."""
    if label.optimal_steps <= 0:
        return EvalResult(
            trace_id=trace.trace_id, layer="end_to_end",
            metric="efficiency_score", score=0.5,
            explanation="No optimal steps label provided."
        )

    actual = trace.total_steps
    if actual == 0:
        return EvalResult(
            trace_id=trace.trace_id, layer="end_to_end",
            metric="efficiency_score", score=0.0,
            explanation="Zero steps taken."
        )

    ratio = label.optimal_steps / actual
    score = max(0.0, min(1.0, ratio))

    return EvalResult(
        trace_id=trace.trace_id, layer="end_to_end",
        metric="efficiency_score", score=score,
        explanation=f"Optimal: {label.optimal_steps}, Actual: {actual}, Ratio: {score:.2f}",
        details={"optimal": label.optimal_steps, "actual": actual},
    )


@register(name="hallucination_check", layer="end_to_end", metric="hallucination_check")
def eval_hallucination(trace: Trace, label: Label) -> EvalResult:
    """LLM-as-judge: check the final output for fabricated facts or unsupported claims."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a fact-checker evaluating an AI agent's output for hallucinations. "
                "A hallucination is a claim that is fabricated, unsupported, or contradicts "
                "known facts. Score from 0.0 (severe hallucinations) to 1.0 (no hallucinations). "
                "Respond with JSON: {\"score\": <float>, \"hallucinations\": [\"<claim>\"], \"explanation\": \"<brief>\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {trace.input}\n\nAgent final output:\n{trace.final_output}"
            ),
        },
    ]

    try:
        result = chat_json(messages, temperature=0.0)
        score = float(result.get("score", 1.0))
        explanation = result.get("explanation", "LLM judge evaluation.")
        score = max(0.0, min(1.0, score))
    except Exception as e:
        score = 1.0
        explanation = f"LLM judge evaluation failed: {e}"

    return EvalResult(
        trace_id=trace.trace_id, layer="end_to_end",
        metric="hallucination_check", score=score,
        explanation=explanation,
    )
