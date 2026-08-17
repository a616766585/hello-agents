import unittest

from ReAct.ReActAgent import ReActAgent
from ReAct.ToolExecutor import ToolExecutor


class FakeLLM:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.prompts = []

    def think(self, messages):
        self.prompts.append(messages[0]["content"])
        return next(self._responses)


class ReActAgentTest(unittest.TestCase):
    def setUp(self):
        self.tools = ToolExecutor()
        self.calls = []
        self.tools.registerTool("Search", "searches current information", self._search)

    def _search(self, query):
        self.calls.append(query)
        return f"结果：{query}"

    def test_runs_action_then_returns_final_answer(self):
        llm = FakeLLM(
            [
                "Thought: 需要查找。\nAction: Search\nAction Input: Python 3.14",
                "Thought: 信息足够。\nFinal Answer: Python 3.14 是测试结果。",
            ]
        )

        answer = ReActAgent(llm, self.tools, verbose=False).run("Python 3.14 是什么？")

        self.assertEqual("Python 3.14 是测试结果。", answer)
        self.assertEqual(["Python 3.14"], self.calls)
        self.assertIn("Observation: 结果：Python 3.14", llm.prompts[1])

    def test_unknown_tool_becomes_an_observation(self):
        llm = FakeLLM(
            [
                "Action: Missing\nAction Input: anything",
                "Final Answer: 已根据工具错误调整方案。",
            ]
        )
        agent = ReActAgent(llm, self.tools, verbose=False)

        self.assertEqual("已根据工具错误调整方案。", agent.run("测试"))
        self.assertIn("未找到名为 'Missing' 的工具", llm.prompts[1])

    def test_parser_accepts_compact_action_syntax(self):
        self.assertEqual(("", "Search", "hello"), ReActAgent.parse_response("Action: Search[hello]"))


if __name__ == "__main__":
    unittest.main()
