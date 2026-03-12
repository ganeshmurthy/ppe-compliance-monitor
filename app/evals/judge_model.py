"""Custom DeepEval judge model backed by the same VLLM endpoint the app uses."""

import os

from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM


class VLLMJudge(DeepEvalBaseLLM):
    """Wraps the project's OpenAI-compatible VLLM endpoint for use as a
    DeepEval LLM-as-a-judge evaluator.

    Reads connection details from the same env vars as the application:
    ``OPENAI_API_ENDPOINT``, ``OPENAI_API_TOKEN``, ``OPENAI_MODEL``.
    """

    def __init__(self) -> None:
        self._chat = ChatOpenAI(
            base_url=os.environ["OPENAI_API_ENDPOINT"],
            api_key=os.environ["OPENAI_API_TOKEN"],
            model=os.getenv("OPENAI_MODEL", "llama-4-scout-17b-16e-w4a16"),
            temperature=0.7,
        )

    def load_model(self):
        return self._chat

    def generate(self, prompt: str) -> str:
        return self._chat.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self._chat.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return "VLLM Judge"
