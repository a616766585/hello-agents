"""Run the ReAct search example from the repository root.

    python -m ReAct.Main
"""

from common.llm import HelloAgentsLLM
from ReAct.ReActAgent import ReActAgent
from ReAct.Search import search
from ReAct.ToolExecutor import ToolExecutor


def build_agent() -> ReActAgent:
    """Create the tutorial agent with the web-search tool registered."""
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        "Search",
        "一个网页搜索引擎。当你需要回答时事、事实或知识库中没有的信息时，应使用此工具。",
        search,
    )
    return ReActAgent(HelloAgentsLLM(), tool_executor, max_steps=5)


if __name__ == "__main__":
    try:
        agent = build_agent()
        question = "英伟达最新的 GPU 型号是什么？"
        print(f"\n--- 用户问题 ---\n{question}")
        answer = agent.run(question)
        print(f"\n--- ReAct 最终回答 ---\n{answer}")
    except ValueError as exc:
        # HelloAgentsLLM uses this error when its .env configuration is absent.
        print(f"初始化失败：{exc}")
