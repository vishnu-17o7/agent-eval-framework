"""OpenRouter LLM client for agent inference and model-based evals.

Supports two modes:
- LIVE: when OPENROUTER_API_KEY is set, makes real API calls
- MOCK: when key is not set, uses simulated responses for demo/testing
"""

import os
import json
import hashlib
import httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

_MOCK_MODE = not os.environ.get("OPENROUTER_API_KEY", "")


def _get_api_key() -> str:
    if _MOCK_MODE:
        return "mock-key"
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it with: $env:OPENROUTER_API_KEY='your-key'  (PowerShell)"
        )
    return key


def _mock_chat(messages: list[dict]) -> str:
    """Generate a mock chat response based on the last user message content.

    Simulates agent reasoning + tool calls or final answers.
    """
    last_user = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m["content"]
            break

    query_lower = last_user.lower()

    if "tool result from search" in query_lower or "tool result from read_file" in query_lower:
        return "THOUGHT: I have received the search results. The information is relevant and addresses the user's query. I can now provide a final answer.\nANSWER: Based on the search results, I found relevant information. The topic is well-documented and the key points have been covered in the data retrieved."

    if "tool result from summarize" in query_lower:
        return "THOUGHT: The summary has been generated successfully. I can now present the condensed information to the user.\nANSWER: Here is the summarized information as requested."

    if "Please provide your final answer" in query_lower:
        return "THOUGHT: I need to provide my final synthesis based on what I've gathered.\nANSWER: Based on the information collected, I can confirm the key findings. The data supports a clear answer to the original query."

    if "Please use the TOOL:" in query_lower:
        return generate_mock_agent_response(query_lower)

    return generate_mock_agent_response(last_user)


def generate_mock_agent_response(query: str) -> str:
    """Generate a simulated agent response with THOUGHT/TOOL/PARAMS or ANSWER."""
    query_lower = query.lower()

    if "summarize" in query_lower:
        if "search" in query_lower or "find" in query_lower:
            return 'THOUGHT: The user wants me to search for information first, then summarize. I should start with a search.\nTOOL: search\nPARAMS: {"query": "the requested topic"}'
        return 'THOUGHT: The user wants me to summarize information. I will generate a concise summary.\nTOOL: summarize\nPARAMS: {"text": "the content to summarize"}'

    if "read" in query_lower or "file" in query_lower:
        return 'THOUGHT: The user wants me to read a file. I should use the read_file tool.\nTOOL: read_file\nPARAMS: {"path": "the requested file path"}'

    return 'THOUGHT: The user is asking for factual information. I should search for this topic to get accurate data.\nTOOL: search\nPARAMS: {"query": "the requested topic"}'


def _mock_chat_json(messages: list[dict]) -> dict:
    """Generate a mock JSON response for eval judgments.

    Uses hash of the last user message to produce varied but deterministic scores.
    """
    last_user = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user = m["content"]
            break

    h = int(hashlib.md5(last_user.encode()).hexdigest()[:8], 16)
    score = 0.55 + (h % 40) / 100.0
    score = min(1.0, max(0.3, score))

    return {
        "score": score,
        "explanation": f"Mock evaluation: score {score:.2f} based on simulated judgment of the agent's performance.",
    }


def _mock_embedding(text: str) -> list[float]:
    """Generate a mock embedding vector using hash-based pseudo-random values."""
    h = hashlib.sha256(text.encode()).digest()
    return [((b / 255.0) * 2 - 1) for b in h[:64]]


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request to OpenRouter and return the response text."""
    if _MOCK_MODE:
        return _mock_chat(messages)

    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"]


def chat_json(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict:
    """Send a chat completion and parse the response as JSON."""
    if _MOCK_MODE:
        return _mock_chat_json(messages)

    raw = chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw}


def get_embedding(text: str, model: str = "openai/text-embedding-3-small") -> list[float]:
    """Get an embedding vector for the given text via OpenRouter."""
    if _MOCK_MODE:
        return _mock_embedding(text)

    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": text,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{OPENROUTER_BASE}/embeddings",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return data["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
