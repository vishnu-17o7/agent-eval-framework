"""Sample Research Assistant agent — two versions for evaluation."""

import json
import re
from agent_eval.core.llm import chat, chat_json
from agent_eval.agent.tools import TOOLS, execute_tool, TOOL_SCHEMAS
from agent_eval.tracing.collector import TraceCollector


SYSTEM_PROMPT_V1 = """You are a research assistant. You can use tools to help answer questions.
Available tools: search, read_file, summarize.

When you need to use a tool, respond in this exact format:

THOUGHT: <your reasoning about what to do next>
TOOL: <tool_name>
PARAMS: <JSON parameters>

If you have the final answer, respond:

THOUGHT: <final reasoning>
ANSWER: <your final answer>

Always think step by step before calling a tool."""


SYSTEM_PROMPT_V2 = """You are an expert research assistant with strong analytical skills.
You have access to these tools:

- **search(query: str)**: Search the web for information. Use when you need facts or data.
- **read_file(path: str)**: Read a file from the local filesystem. Use for local documents.
- **summarize(text: str)**: Generate a concise summary of long text.

**Instructions:**
1. Analyze the user's question carefully. Identify what information is needed.
2. Select the most appropriate tool for the task:
   - If the user asks about a specific file, use read_file first.
   - If the user asks for factual information, use search.
   - If you have a long result and need to condense it, use summarize.
3. Before each tool call, write your reasoning in THOUGHT.
4. After receiving a tool result, verify it's relevant before proceeding.
5. Do NOT repeat the same tool call if it didn't work — try a different approach.
6. When you have enough information, write ANSWER.

**Response format:**

THOUGHT: <your reasoning>
TOOL: <tool_name>
PARAMS: <JSON parameters>

OR

THOUGHT: <final reasoning>
ANSWER: <your final answer>"""


class ResearchAgent:
    """A research assistant agent that uses tools to answer questions."""

    def __init__(self, version: str = "v1"):
        self.version = version
        self.system_prompt = SYSTEM_PROMPT_V2 if version == "v2" else SYSTEM_PROMPT_V1
        self.max_steps = 8

    def run(self, query: str, collector: TraceCollector | None = None) -> str:
        """Run the agent on a query and return the final answer.

        If a TraceCollector is provided, traces are captured automatically.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        if collector:
            collector.start_run(self.version, query)

        for step_num in range(1, self.max_steps + 1):
            response = chat(messages)

            thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\nTOOL:|\nANSWER:|$)", response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""

            if "ANSWER:" in response:
                answer_match = re.search(r"ANSWER:\s*(.+)", response, re.DOTALL)
                answer = answer_match.group(1).strip() if answer_match else response

                if collector:
                    collector.add_step(
                        step_number=step_num,
                        reasoning=thought,
                        intent=self._extract_intent(thought),
                        selected_tool=None,
                        tool_params=None,
                        tool_result=None,
                        next_action="answer",
                    )
                    collector.finish_run(answer)

                return answer

            tool_match = re.search(r"TOOL:\s*(\S+)", response)
            params_match = re.search(r"PARAMS:\s*(.+?)(?=\n|$)", response, re.DOTALL)

            if not tool_match:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please use the TOOL: and PARAMS: format to call a tool, or ANSWER: to give the final answer.",
                })
                continue

            tool_name = tool_match.group(1).strip()
            params_str = params_match.group(1).strip() if params_match else "{}"

            try:
                params = json.loads(params_str) if params_str else {}
            except json.JSONDecodeError:
                params = {"query": params_str}

            tool_result = execute_tool(tool_name, params)
            intent = self._extract_intent(thought)

            if collector:
                collector.add_step(
                    step_number=step_num,
                    reasoning=thought,
                    intent=intent,
                    selected_tool=tool_name,
                    tool_params=params,
                    tool_result=str(tool_result),
                    next_action="continue",
                )

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result from {tool_name}({json.dumps(params)}):\n{tool_result}",
            })

        final_response = chat(messages + [
            {"role": "user", "content": "Please provide your final answer now."}
        ])

        if collector:
            collector.add_step(
                step_number=self.max_steps + 1,
                reasoning="Max steps reached, providing final answer.",
                intent="conclude",
                selected_tool=None,
                tool_params=None,
                tool_result=None,
                next_action="answer",
            )
            collector.finish_run(final_response)

        return final_response

    def _extract_intent(self, thought: str) -> str:
        """Heuristic to extract user intent from the agent's reasoning."""
        thought_lower = thought.lower()
        if "search" in thought_lower or "find" in thought_lower or "look up" in thought_lower:
            return "search_information"
        if "read" in thought_lower or "file" in thought_lower or "open" in thought_lower:
            return "read_file"
        if "summarize" in thought_lower or "summar" in thought_lower or "condense" in thought_lower:
            return "summarize_text"
        return "unknown"
