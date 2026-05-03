"""Layer 1 — Pre-Tool-Call evals (reasoning quality before tool execution)."""

from agent_eval.core.types import Trace, Label, EvalResult
from agent_eval.core.registry import register
from agent_eval.core.llm import get_embedding, cosine_similarity, chat_json


@register(name="intent_accuracy", layer="pre_tool", metric="intent_accuracy")
def eval_intent_accuracy(trace: Trace, label: Label) -> EvalResult:
    """Measure how well the agent's identified intent matches the expected intent.

    Uses OpenRouter embedding similarity between the agent's stated intent
    and the labeled expected intent.
    """
    if not label.expected_intent:
        return EvalResult(
            trace_id=trace.trace_id, layer="pre_tool",
            metric="intent_accuracy", score=0.5,
            explanation="No expected intent label provided."
        )

    first_step = trace.steps[0] if trace.steps else None
    if first_step is None:
        return EvalResult(
            trace_id=trace.trace_id, layer="pre_tool",
            metric="intent_accuracy", score=0.0,
            explanation="No steps in trace."
        )

    agent_intent = first_step.intent or "unknown"
    try:
        emb_agent = get_embedding(agent_intent)
        emb_expected = get_embedding(label.expected_intent)
        similarity = cosine_similarity(emb_agent, emb_expected)
        score = max(0.0, min(1.0, similarity))
    except Exception as e:
        score = 0.5
        return EvalResult(
            trace_id=trace.trace_id, layer="pre_tool",
            metric="intent_accuracy", score=score,
            explanation=f"Embedding comparison failed: {e}"
        )

    return EvalResult(
        trace_id=trace.trace_id, layer="pre_tool",
        metric="intent_accuracy", score=score,
        explanation=f"Agent intent: '{agent_intent}' vs expected: '{label.expected_intent}'. Similarity: {score:.3f}",
        details={"agent_intent": agent_intent, "expected_intent": label.expected_intent, "similarity": score},
    )


@register(name="tool_selection", layer="pre_tool", metric="tool_selection")
def eval_tool_selection(trace: Trace, label: Label) -> EvalResult:
    """Check if the agent selected the correct tool for the task.

    Deterministic: exact match against labeled expected tool.
    """
    if not label.expected_tool:
        return EvalResult(
            trace_id=trace.trace_id, layer="pre_tool",
            metric="tool_selection", score=0.5,
            explanation="No expected tool label provided."
        )

    first_tool_step = next((s for s in trace.steps if s.selected_tool), None)
    if first_tool_step is None:
        return EvalResult(
            trace_id=trace.trace_id, layer="pre_tool",
            metric="tool_selection", score=0.0,
            explanation="Agent made no tool calls."
        )

    selected = first_tool_step.selected_tool
    expected = label.expected_tool
    score = 1.0 if selected == expected else 0.0

    return EvalResult(
        trace_id=trace.trace_id, layer="pre_tool",
        metric="tool_selection", score=score,
        explanation=f"Selected '{selected}', expected '{expected}'. {'Match' if score > 0 else 'Mismatch'}.",
        details={"selected": selected, "expected": expected},
    )


@register(name="reasoning_quality", layer="pre_tool", metric="reasoning_quality")
def eval_reasoning_quality(trace: Trace, label: Label) -> EvalResult:
    """LLM-as-judge evaluation of the agent's reasoning quality.

    Scores reasoning on clarity, relevance, and logical soundness.
    """
    all_reasoning = "\n---\n".join(
        f"Step {s.step_number}: {s.reasoning}"
        for s in trace.steps if s.reasoning
    )

    if not all_reasoning.strip():
        return EvalResult(
            trace_id=trace.trace_id, layer="pre_tool",
            metric="reasoning_quality", score=0.0,
            explanation="No reasoning found in trace."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator of AI agent reasoning. "
                "Score the agent's chain-of-thought reasoning on a scale of 0.0 to 1.0 "
                "based on three criteria: clarity (is it easy to follow?), "
                "relevance (does it directly address the user's query?), and "
                "logical soundness (are the conclusions well-supported?). "
                "Respond with a JSON object: {\"score\": <float>, \"explanation\": \"<brief>\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {trace.input}\n\nAgent reasoning:\n{all_reasoning}"
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
        trace_id=trace.trace_id, layer="pre_tool",
        metric="reasoning_quality", score=score,
        explanation=explanation,
        details={"raw_result": result if 'result' in dir() else {}},
    )
