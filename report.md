# Agent Evaluation Framework — Full Report

## Overview

AI agents that use tools (search, code execution, database queries, API calls) are increasingly deployed in production. Evaluating them is hard: the final output only tells you whether the agent succeeded or failed, not *where* it went wrong. An agent could pick the right answer for the wrong reasons, or fail at step 3 out of 7 in a way that is invisible from the final output alone.

This framework addresses that gap by measuring agent quality at every stage of execution — not just the final output. It is framework-agnostic, supports both deterministic and model-based evals, and produces structured, queryable traces that feed into versioned evaluation datasets and a score history dashboard.

---

## Architecture

```
agent_eval/
├── core/                  # Framework engine
│   ├── runner.py          # EvalRunner: orchestrates evals against agent runs
│   ├── registry.py        # EvalRegistry: discover & register evals via decorators
│   ├── types.py           # Trace, TraceStep, EvalResult, Label, Scenario dataclasses
│   ├── scorer.py          # Aggregate scores across layers (weighted mean)
│   └── llm.py             # OpenRouter client (chat, chat_json, embeddings) + mock mode
├── evals/                 # Eval definitions across three layers
│   ├── pre_tool/          # Layer 1: before tool calls
│   │   └── evals.py       # intent_accuracy, tool_selection, reasoning_quality
│   ├── at_tool/           # Layer 2: at/after tool calls
│   │   └── evals.py       # param_schema_valid, param_correctness, result_interpretation
│   └── end_to_end/        # Layer 3: full run
│       └── evals.py       # goal_completion, efficiency_score, hallucination_check
├── tracing/               # Trace capture & storage
│   ├── collector.py       # Intercepts agent steps into structured Trace objects
│   ├── store.py           # SQLite persistence (traces, scores, eval_results tables)
│   └── query.py           # Query API for filtering traces by layer
├── datasets/              # Dataset management
│   ├── manager.py         # CRUD for scenarios, traces, labels (versioned JSON files)
│   ├── labeler.py         # Interactive CLI for manual trace labeling
│   ├── auto_label.py      # Predefined label map for quick demo setup
│   └── scenarios/         # Versioned JSON dataset files
│       ├── v1/            # traces.json, labels.json, scenarios.json
│       └── v2/            # traces.json, labels.json
├── agent/                 # Sample Research Assistant agent
│   ├── agent.py           # Agent implementation (v1 + v2)
│   ├── tools.py           # Tool definitions (search, read_file, summarize)
│   └── prompts.py         # System prompts (v1 simpler, v2 improved with few-shot)
├── dashboard/             # Reporting
│   └── app.py             # Streamlit dashboard (score comparison, bar charts, gate status)
└── cli.py                 # CLI entry point (6 commands)
```

### Data Model

```
TRACE ────────────── contains ───> TRACE_STEP[]
TRACE ────────────── scored_by ──> EVAL_RESULT[]
DATASET ──────────── references ─> TRACE[]
DATASET ──────────── has ────────> LABEL[]

Trace:
  trace_id (uuid), run_id, agent_version, scenario_id, timestamp
  input (user query), final_output, total_steps
  steps: [TraceStep, ...]

TraceStep:
  step_number, reasoning, intent, selected_tool
  tool_params (dict), tool_result (string), next_action

EvalResult:
  eval_id (uuid), trace_id, layer ("pre_tool"|"at_tool"|"end_to_end")
  metric (string), score (float 0-1), explanation, details

Label (ground truth):
  label_id, trace_id, expected_intent, expected_tool
  expected_params, goal_achieved (bool), optimal_steps

Scenario:
  scenario_id, query, description
```

### LLM Provider — OpenRouter

| Use Case | Model | Notes |
|----------|-------|-------|
| Agent (v1 & v2) | `deepseek/deepseek-v4-flash` | Fast, cost-effective reasoning |
| LLM-as-judge evals | `deepseek/deepseek-v4-flash` | Structured rubric output |
| Embeddings (intent similarity) | `openai/text-embedding-3-small` | Cheap OpenRouter embeddings endpoint |

**Mock mode**: When `OPENROUTER_API_KEY` is not set, the framework falls back to deterministic mock responses (hash-based), enabling full pipeline testing without API costs. Set the env var to switch to live calls.

---

## Three Evaluation Layers

### Layer 1 — Pre-Tool-Call (Reasoning Quality)

Evaluates the agent's reasoning *before* it executes a tool call.

| Eval | Type | What It Measures | How |
|------|------|------------------|-----|
| `intent_accuracy` | Model-based | Does the agent's identified intent match the labeled expected intent? | OpenRouter embedding cosine similarity |
| `tool_selection` | Deterministic | Did the agent select the correct tool for the task? | Exact string match against label |
| `reasoning_quality` | Model-based | Is the agent's chain-of-thought clear, relevant, and logically sound? | LLM-as-judge scoring on a 0-1 rubric |

### Layer 2 — At/After Tool Call (Execution Quality)

Evaluates the agent's tool usage — parameters and result interpretation.

| Eval | Type | What It Measures | How |
|------|------|------------------|-----|
| `param_schema_valid` | Deterministic | Do tool parameters conform to the tool's JSON Schema? | `jsonschema.validate()` against `TOOL_SCHEMAS` |
| `param_correctness` | Model-based | Are parameter values semantically correct for the intent? | LLM-as-judge comparing actual vs expected params |
| `result_interpretation` | Model-based | Did the agent correctly interpret the tool result and decide the next step? | LLM-as-judge evaluating next-action choice |

### Layer 3 — End-to-End

Evaluates the complete agent run holistically.

| Eval | Type | What It Measures | How |
|------|------|------------------|-----|
| `goal_completion` | Model-based | Did the agent accomplish the user's goal? | LLM-as-judge: binary 0/1 with explanation |
| `efficiency_score` | Deterministic | How efficient was the agent? | `optimal_steps / actual_steps` ratio |
| `hallucination_check` | Model-based | Are there fabricated facts or unsupported claims in the final output? | LLM-as-judge fact-checking the output |

---

## Sample Agent: Research Assistant

The framework ships with a Research Assistant agent built on OpenRouter `deepseek/deepseek-v4-flash`. It has access to three tools:

- **search(query: str)** — Simulated web search against a knowledge base of 8 topics (climate change, Python, machine learning, electric cars, space exploration, quantum computing, renewable energy, AI)
- **read_file(path: str)** — Read local file contents
- **summarize(text: str)** — Generate a concise summary

The agent follows a structured THOUGHT → TOOL → PARAMS / ANSWER loop, making it straightforward to parse and evaluate at each step.

### Two Agent Versions

**v1** — Basic system prompt. Linear reasoning. Prone to tool selection errors and repetitive calls.

**v2** — Improved prompt with:
- Explicit tool descriptions and usage guidance
- Few-shot examples of correct tool selection
- "Verify before proceeding" instruction
- Clear fallback strategy (don't repeat failed calls)

---

## Evaluation Dataset

### Scenarios (12)

| ID | Query | Type |
|----|-------|------|
| s01 | What is climate change and what causes it? | Factual lookup |
| s02 | Tell me about the Python programming language. | Factual lookup |
| s03 | Search for information about electric cars. | Web search |
| s04 | What is machine learning? Give me a detailed explanation. | Detailed knowledge |
| s05 | Look up information about space exploration milestones. | Factual lookup |
| s06 | Search for renewable energy and summarize what you find. | Search + summarize |
| s07 | What can you tell me about quantum computing? | Knowledge lookup |
| s08 | Find information about artificial intelligence and summarize it. | Search + summarize |
| s09 | Tell me about climate change and then summarize the key points. | Multi-step |
| s10 | Search for electric cars and then explain their benefits. | Search + explain |
| s11 | What is Python? Also tell me about machine learning. | Multi-topic |
| s12 | Find information about space exploration and summarize the highlights. | Search + summarize |

### Labels (Ground Truth)

Each scenario has a predefined label specifying:
- **expected_intent** — The correct intent the agent should identify (e.g., `search_information`)
- **expected_tool** — The correct tool to use (e.g., `search`)
- **expected_params** — The semantically correct parameters (e.g., `{"query": "climate change"}`)
- **goal_achieved** — Whether the task is achievable with available tools
- **optimal_steps** — Minimum expected steps (1-4 depending on complexity)

---

## Results

### Score Comparison (v1 vs v2)

Both versions were run on the same 12 scenarios with the same labels. In mock mode, scores are identical because mock responses are deterministic per-input. With a live API key, v2 would typically score higher on `reasoning_quality` and `tool_selection` due to its improved prompt.

```
======================================================================
  SCORE HISTORY
======================================================================
  Version     Overall Pre-Tool  At-Tool      E2E  Timestamp
----------------------------------------------------------------------
  v1            0.877    0.866    0.865    0.901  2026-05-04T01:00:52
  v2            0.877    0.866    0.865    0.901  2026-05-04T01:00:54
  v1            0.877    0.866    0.865    0.901  2026-05-04T02:21:31
  v2            0.877    0.866    0.865    0.901  2026-05-04T02:21:33
======================================================================
```

### Per-Metric Breakdown

| Metric | Layer | Score | Type |
|--------|-------|-------|------|
| intent_accuracy | pre_tool | 0.917 | Model-based |
| tool_selection | pre_tool | 0.917 | Deterministic |
| reasoning_quality | pre_tool | 0.763 | Model-based |
| param_schema_valid | at_tool | 1.000 | Deterministic |
| param_correctness | at_tool | 0.788 | Model-based |
| result_interpretation | at_tool | 0.808 | Model-based |
| goal_completion | end_to_end | 1.000 | Model-based |
| efficiency_score | end_to_end | 1.000 | Deterministic |
| hallucination_check | end_to_end | 0.704 | Model-based |

### Gate Status

Threshold: **0.60** — Both v1 and v2 **PASSED**.

---

## Usage Guide

### Installation

```bash
pip install httpx jsonschema streamlit sqlite-utils
```

### Commands

```bash
# Generate scenarios and run agent v1 on all 12, capturing traces
python -m agent_eval.cli run-scenarios --version v1 --generate

# Run agent v2
python -m agent_eval.cli run-scenarios --version v2

# Auto-generate labels (no manual labeling needed)
python -m agent_eval.cli auto-label --version v1
python -m agent_eval.cli auto-label --version v2

# Run all 9 evals against labeled traces
python -m agent_eval.cli eval --version v1
python -m agent_eval.cli eval --version v2 --gate 0.6

# View score history report
python -m agent_eval.cli report

# Launch Streamlit dashboard
python -m agent_eval.cli dashboard --port 8501
```

### Using a Live API Key

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
```

Without the key, the framework runs in mock mode with deterministic simulated responses — perfect for testing and demos. Set the key to use real LLM calls through OpenRouter.

### Adding New Evals

New evals are registered via the `@register` decorator. Example:

```python
from agent_eval.core.registry import register
from agent_eval.core.types import Trace, Label, EvalResult

@register(name="my_custom_eval", layer="pre_tool", metric="custom_metric")
def my_eval(trace: Trace, label: Label) -> EvalResult:
    score = 0.85  # your evaluation logic
    return EvalResult(
        trace_id=trace.trace_id,
        layer="pre_tool",
        metric="custom_metric",
        score=score,
        explanation="Custom evaluation result."
    )
```

The eval is automatically discovered and included in all future `eval` runs — no modification to the agent or runner needed.

### Adding New Tools

Add a new tool schema to `agent/tools.py`:

```python
TOOL_SCHEMAS["new_tool"] = {
    "type": "object",
    "properties": {
        "param1": {"type": "string"},
    },
    "required": ["param1"],
}

def execute_tool(tool_name, params):
    # ... add case for "new_tool"
```

The `param_schema_valid` eval automatically picks up the new schema.

---

## Design Decisions

### Why SQLite + JSON for storage?
Lightweight, zero-configuration, queryable via SQL, and human-readable JSON files for dataset versioning. No need for external databases or services.

### Why registration-based evals?
The `@register` decorator pattern means new evals can be added by simply importing a new file — the runner discovers them automatically. This satisfies the requirement that evals be defined without modifying the agent itself.

### Why mock mode?
Enables immediate testing and demonstration without API costs or key management. The mock responses are deterministic (hash-based), making behavior reproducible. Switching to live mode requires only setting one environment variable.

### Why OpenRouter?
Single provider for both chat and embeddings. The `deepseek/deepseek-v4-flash` model balances speed, quality, and cost. OpenRouter's unified API means no provider-specific code changes are needed to switch models.

### Integration inspiration
The tracing model draws from **LangSmith** (structured step-level traces with metadata). The dataset management draws from **Braintrust** (versioned datasets with labels that evolve over time). The LLM-as-judge pattern aligns with **RAGAS** and **PromptFoo** approaches to model-based evaluation.

---

## Technology Stack

| Component | Choice | Version |
|-----------|--------|---------|
| Language | Python | 3.12.4 |
| LLM Provider | OpenRouter | `deepseek/deepseek-v4-flash` |
| Embeddings | OpenRouter | `openai/text-embedding-3-small` |
| HTTP Client | httpx | 0.28.1 |
| Schema Validation | jsonschema | 4.26.0 |
| Trace Storage | SQLite + JSON | stdlib |
| Dataset Versioning | Git-tracked directories | 2.53.0 |
| Dashboard | Streamlit | 1.44.1 |
| CLI | argparse (stdlib) | — |

---

## File Listing

```
pyproject.toml
agent_eval/
  __init__.py
  cli.py
  core/
    __init__.py
    types.py           # Trace, TraceStep, EvalResult, Label, Scenario
    llm.py             # OpenRouter client + mock mode
    registry.py        # @register decorator + EvalRegistry
    runner.py          # EvalRunner: orchestrate evals
    scorer.py          # Score aggregation (per-layer, overall)
  evals/
    __init__.py        # Imports all three layers
    pre_tool/
      __init__.py
      evals.py         # intent_accuracy, tool_selection, reasoning_quality
    at_tool/
      __init__.py
      evals.py         # param_schema_valid, param_correctness, result_interpretation
    end_to_end/
      __init__.py
      evals.py         # goal_completion, efficiency_score, hallucination_check
  tracing/
    __init__.py
    collector.py       # TraceCollector: intercept agent steps
    store.py           # SQLite persistence (traces, scores, eval_results)
    query.py           # Filter traces by layer
  datasets/
    __init__.py
    manager.py         # DatasetManager: CRUD for scenarios/traces/labels
    labeler.py         # Interactive CLI labeling
    auto_label.py      # Predefined label map
    scenarios/
      v1/
        scenarios.json # 12 scenario definitions
        traces.json    # 12 agent traces (v1)
        labels.json    # 12 ground-truth labels
      v2/
        traces.json    # 12 agent traces (v2)
        labels.json    # 12 ground-truth labels
  agent/
    __init__.py
    agent.py           # ResearchAgent with v1 and v2 support
    tools.py           # search, read_file, summarize + schemas
    prompts.py         # System prompts for v1 and v2
  dashboard/
    __init__.py
    app.py             # Streamlit dashboard (score comparison, charts, gate status)
```

---

## Future Extensions

- **CI/CD integration**: GitHub Actions workflow that runs `agent-eval eval` on every PR and posts results as a comment
- **More agent frameworks**: Add adapters for LangChain, CrewAI, and Bedrock Agent Core traces
- **Human annotation UI**: Web-based labeling interface instead of CLI
- **Statistical analysis**: Confidence intervals on scores, A/B testing between versions
- **Alerting**: Slack/webhook notifications when scores drop below threshold
