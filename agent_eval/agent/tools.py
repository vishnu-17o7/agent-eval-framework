"""Tool definitions for the Research Assistant agent."""

import json
import os

# Simulated knowledge base for search tool
_KNOWLEDGE_BASE = {
    "climate change": "Climate change refers to long-term shifts in temperatures and weather patterns. "
        "Human activities have been the main driver since the 1800s, primarily due to burning fossil fuels. "
        "Key effects include rising global temperatures, melting ice caps, and extreme weather events.",
    "python": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. "
        "It emphasizes code readability and supports multiple programming paradigms including procedural, "
        "object-oriented, and functional programming.",
    "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn "
        "and improve from experience without being explicitly programmed. Common approaches include supervised, "
        "unsupervised, and reinforcement learning.",
    "electric cars": "Electric vehicles (EVs) use electric motors powered by rechargeable battery packs. "
        "They produce zero tailpipe emissions and have lower operating costs than gasoline vehicles. "
        "Major manufacturers include Tesla, BYD, and Volkswagen.",
    "space exploration": "Space exploration is the discovery and study of celestial structures using "
        "space technology. Key milestones include the Apollo moon landings (1969), Mars rover missions, "
        "and the James Webb Space Telescope launched in 2021.",
    "quantum computing": "Quantum computing uses quantum bits (qubits) that can exist in multiple states "
        "simultaneously through superposition. This enables certain computations to be performed "
        "exponentially faster than classical computers for specific problems.",
    "renewable energy": "Renewable energy comes from sources that are naturally replenished, such as "
        "sunlight, wind, water, and geothermal heat. Solar and wind power are the fastest-growing "
        "renewable energy sources globally.",
    "artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. "
        "It encompasses machine learning, natural language processing, computer vision, and robotics. "
        "Recent advances in large language models have dramatically expanded AI capabilities.",
}

TOOL_SCHEMAS = {
    "search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"],
    },
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"}
        },
        "required": ["path"],
    },
    "summarize": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to summarize"}
        },
        "required": ["text"],
    },
}

TOOLS = list(TOOL_SCHEMAS.keys())


def execute_tool(tool_name: str, params: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "search":
        query = params.get("query", "").lower()
        for key, value in _KNOWLEDGE_BASE.items():
            if key in query:
                return value
        return f"No results found for query: {params.get('query', '')}"

    elif tool_name == "read_file":
        path = params.get("path", "")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return content[:2000]
        return f"File not found: {path}"

    elif tool_name == "summarize":
        text = params.get("text", "")
        if len(text) < 50:
            return text
        sentences = text.split(". ")
        if len(sentences) <= 2:
            return text
        return ". ".join(sentences[:2]) + "."

    else:
        return f"Unknown tool: {tool_name}"
