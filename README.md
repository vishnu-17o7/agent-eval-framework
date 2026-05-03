# Agent Evaluation Framework

A multi-layer evaluation framework for AI agents that measures quality at every stage of execution — not just the final output.

## Overview

AI agents that use tools (search, code execution, API calls) are hard to evaluate. The final output only tells you success or failure, not *where* it went wrong. This framework solves that by evaluating agents across **three layers**:

| Layer | When | What It Measures |
|-------|------|------------------|
| **Pre-Tool** | Before a tool call | Intent accuracy, tool selection, reasoning quality |
| **At-Tool** | At/after a tool call | Parameter schema validity, parameter correctness, result interpretation |
| **End-to-End** | Full agent run | Goal completion, efficiency, hallucination detection |

**9 evals total** — 4 deterministic + 5 LLM-as-judge (model-based).

## Quick Start

```bash
# 1. Install dependencies
pip install httpx jsonschema streamlit sqlite-utils

# 2. Run agent v1 on 12 scenarios (generates traces)
python -m agent_eval.cli run-scenarios --version v1 --generate

# 3. Run agent v2
python -m agent_eval.cli run-scenarios --version v2

# 4. Auto-label traces (no manual labeling needed)
python -m agent_eval.cli auto-label --version v1
python -m agent_eval.cli auto-label --version v2

# 5. Run all 9 evals
python -m agent_eval.cli eval --version v1
python -m agent_eval.cli eval --version v2

# 6. View scores
python -m agent_eval.cli report

# 7. Launch dashboard
python -m streamlit run agent_eval\dashboard\app.py --server.port 8501
```

**Mock mode** — runs without an API key using deterministic simulated responses. To use live OpenRouter, copy `.env.example` to `.env` and add your key.

## Example Output

```
============================================================
  EVAL RESULTS — Agent v1
============================================================
  Overall Score:   0.877
  Pre-Tool:        0.866
  At-Tool:         0.865
  End-to-End:      0.901
------------------------------------------------------------
  intent_accuracy           0.917
  tool_selection            0.917
  reasoning_quality         0.763
  param_schema_valid        1.000
  param_correctness         0.788
  result_interpretation     0.808
  goal_completion           1.000
  efficiency_score          1.000
  hallucination_check       0.704
============================================================

Gate threshold: 0.60 — PASSED
```

## Project Structure

```
agent_eval/
├── core/              # EvalRunner, Registry, Scorer, LLM client
├── evals/
│   ├── pre_tool/      # intent_accuracy, tool_selection, reasoning_quality
│   ├── at_tool/       # param_schema_valid, param_correctness, result_interpretation
│   └── end_to_end/    # goal_completion, efficiency_score, hallucination_check
├── tracing/           # TraceCollector, SQLite store, query API
├── datasets/          # Manager, labeler, auto-label, versioned JSON files
├── agent/             # Research Assistant agent (v1 + v2) with tools
├── dashboard/         # Streamlit dashboard
└── cli.py             # CLI entry point
```

## Key Design Decisions

- **Framework-agnostic** — works with any agent, any framework (LangChain, custom, Bedrock, etc.)
- **Registration-based evals** — add new evals via `@register` decorator, no agent modification needed
- **SQLite + JSON storage** — lightweight, zero-config, queryable
- **Mock mode** — full pipeline runs without API costs; set `OPENROUTER_API_KEY` to switch live
- **Versioned datasets** — traces and labels stored per agent version for regression tracking
- **Gate threshold** — configurable minimum score that can block bad releases

## LLM Provider

| Use Case | Model |
|----------|-------|
| Agent (v1 & v2) | `deepseek/deepseek-v4-flash` |
| LLM-as-judge evals | `deepseek/deepseek-v4-flash` |
| Embeddings | `openai/text-embedding-3-small` |

All via [OpenRouter](https://openrouter.ai/) — single API key, single endpoint.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | No | — | API key for live LLM; mock mode if unset |
| `AGENT_EVAL_DB` | No | `~/.agent_eval.db` | SQLite database path |
| `AGENT_EVAL_DATASET_DIR` | No | `agent_eval/datasets/scenarios/` | Dataset directory |

## Adding Custom Evals

```python
from agent_eval.core.registry import register
from agent_eval.core.types import Trace, Label, EvalResult

@register(name="my_eval", layer="pre_tool", metric="custom_metric")
def my_eval(trace: Trace, label: Label) -> EvalResult:
    return EvalResult(
        trace_id=trace.trace_id,
        layer="pre_tool",
        metric="custom_metric",
        score=0.85,
        explanation="Custom evaluation."
    )
```

The eval is auto-discovered — no changes to the runner or agent needed.

## Full Documentation

See [report.md](report.md) for the complete report covering architecture, data model, design rationale, and full results.
