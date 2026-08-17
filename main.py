"""Backward-compatible import for the shared LLM client.

New modules should import ``HelloAgentsLLM`` from ``common.llm`` instead.
"""

from common.llm import HelloAgentsLLM

__all__ = ["HelloAgentsLLM"]
