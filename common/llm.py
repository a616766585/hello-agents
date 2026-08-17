"""OpenAI-compatible LLM client shared by all agents."""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class HelloAgentsLLM:
    """Call an OpenAI-compatible chat-completions service with streaming."""

    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = apiKey or os.getenv("LLM_API_KEY")
        base_url = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        if not all([self.model, api_key, base_url]):
            raise ValueError("模型 ID、API 密钥和服务地址必须被提供或在 .env 文件中定义。")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> Optional[str]:
        """Send messages to the model and return the complete streamed text."""
        print(f"正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as exc:
            print(f"调用 LLM API 时发生错误: {exc}")
            return None
