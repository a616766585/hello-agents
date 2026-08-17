"""A small, dependency-free implementation of the ReAct agent pattern.

The model is asked to produce either a ``Final Answer`` or an action in the
following form::

    Thought: ...
    Action: ToolName
    Action Input: ...

After a tool is called its result is added to the next prompt as an
``Observation``. Keeping the loop here (rather than in the LLM client) makes
the agent usable with any OpenAI-compatible client and with simple fakes in
tests.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple


REACT_PROMPT_TEMPLATE = """你是一个遵循 ReAct（Reasoning and Acting）模式的智能助手。
请根据问题逐步思考；需要外部信息时调用工具，并严格使用下面的格式输出：

Thought: 对当前问题的分析
Action: 工具名称
Action Input: 传给工具的输入

工具返回结果会以 Observation 的形式提供给你。获得足够信息后，使用：

Final Answer: 给用户的最终答案

可用工具：
{tools}

问题：
{question}

历史思考与观察：
{history}
"""


class ReActAgent:
    """Run a bounded ReAct thought/action/observation loop.

    ``llm_client`` only needs a ``think(messages=...)`` method and
    ``tool_executor`` needs ``getAvailableTools`` and ``getTool``. This loose
    protocol keeps the class easy to use with the project's ``HelloAgentsLLM``
    as well as test doubles.
    """

    def __init__(
        self,
        llm_client: Any,
        tool_executor: Any,
        max_steps: int = 5,
        *,
        verbose: bool = True,
        prompt_template: str = REACT_PROMPT_TEMPLATE,
    ) -> None:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps 必须是大于 0 的整数。")
        if not hasattr(llm_client, "think"):
            raise TypeError("llm_client 必须提供 think(messages=...) 方法。")
        if not hasattr(tool_executor, "getTool"):
            raise TypeError("tool_executor 必须提供 getTool(name) 方法。")

        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.verbose = verbose
        self.prompt_template = prompt_template
        # A transcript of model messages and observations from the latest run.
        self.history: list[str] = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _parse_response(response: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Return ``(final_answer, action, action_input)`` from model text.

        Final answers take precedence over actions. Action input is allowed to
        span multiple lines, which is useful when a model emits JSON or a long
        search query.
        """
        text = response.strip().replace("```text", "").replace("```", "").strip()
        final_match = re.search(r"(?im)^\s*Final\s*Answer\s*:\s*(.*)$", text, re.DOTALL)
        if final_match:
            return final_match.group(1).strip(), None, None

        action_match = re.search(r"(?im)^\s*Action\s*:\s*([^\n]+)", text)
        if not action_match:
            return "", None, None

        action_line = action_match.group(1).strip()
        # Also accept the compact form ``Action: Search[query]``.
        compact_input: Optional[str] = None
        compact_match = re.match(r"^([^\[\s]+)\s*\[(.*)\]\s*$", action_line, re.DOTALL)
        if compact_match:
            action_line, compact_input = compact_match.group(1).strip(), compact_match.group(2).strip()

        input_match = re.search(r"(?im)^\s*Action\s*Input\s*:\s*(.*)$", text, re.DOTALL)
        action_input = compact_input
        if input_match:
            action_input = input_match.group(1).strip()
            # Do not accidentally feed a following label into the tool input.
            action_input = re.split(
                r"\n\s*(?:Observation|Final\s*Answer|Thought|Action)\s*:",
                action_input,
                maxsplit=1,
                flags=re.I,
            )[0].strip()

        return "", action_line, action_input

    @staticmethod
    def parse_response(response: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Public alias useful for callers that want to inspect model output."""
        return ReActAgent._parse_response(response)

    def run(self, question: str) -> str:
        """Answer ``question`` using at most :attr:`max_steps` model calls."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 不能为空。")

        self.history = []
        for step in range(1, self.max_steps + 1):
            self._log(f"--- 第 {step} 步 ---")
            tools_desc = (
                self.tool_executor.getAvailableTools()
                if hasattr(self.tool_executor, "getAvailableTools")
                else "（无可用工具）"
            )
            prompt = self.prompt_template.format(
                tools=tools_desc or "（无可用工具）",
                question=question,
                history="\n".join(self.history) or "（尚无历史记录）",
            )

            try:
                response = self.llm_client.think(messages=[{"role": "user", "content": prompt}])
            except Exception as exc:  # clients may raise instead of returning None
                self._log(f"❌ 调用 LLM 失败: {exc}")
                return f"抱歉，调用语言模型时发生错误：{exc}"

            if not isinstance(response, str) or not response.strip():
                self._log("错误：LLM 未能返回有效响应。")
                return "抱歉，语言模型未返回有效响应。"

            self.history.append(response.strip())
            final_answer, action, action_input = self._parse_response(response)
            if final_answer:
                self._log(f"--- 最终答案 ---\n{final_answer}")
                return final_answer

            if not action or action_input is None:
                observation = "无法解析响应。请严格输出 Action/Action Input，或直接输出 Final Answer。"
                self.history.append(f"Observation: {observation}")
                self._log(f"⚠️ {observation}")
                continue

            try:
                tool: Optional[Callable[[str], Any]] = self.tool_executor.getTool(action)
            except Exception as exc:
                tool = None
                observation = f"获取工具 '{action}' 失败：{exc}"
            else:
                if tool is None:
                    observation = f"未找到名为 '{action}' 的工具。可用工具：{tools_desc or '（无）'}"
                else:
                    try:
                        result = tool(action_input)
                        observation = "" if result is None else str(result)
                    except Exception as exc:
                        observation = f"工具 '{action}' 执行失败：{exc}"

            self.history.append(f"Observation: {observation}")
            self._log(f"--- 观察 (Observation) ---\n{observation}")

        self._log("已达到最大步数，未生成最终答案。")
        return "抱歉，我在规定的最大步骤内未能完成任务。"
