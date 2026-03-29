import asyncio
import os
from typing import Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from tools.mcp_tools import load_tools
from logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a terse workplace safety assistant for a PPE compliance monitoring system. "
    "You have access to read-only PostgreSQL tools via an MCP server. "
    "Use execute_sql to run SELECT queries and answer questions with real data.\n\n"
    "Database schema:\n"
    "  app_config(id SERIAL PK, model_url VARCHAR, model_name VARCHAR, video_source VARCHAR, created_at TIMESTAMP)\n"
    "  detection_classes(id SERIAL PK, app_config_id INTEGER FK→app_config, "
    "model_class_index INTEGER, name VARCHAR, trackable BOOLEAN)\n"
    "  detection_tracks(track_id INTEGER PK, detection_classes_id INTEGER FK→detection_classes, "
    "first_seen TIMESTAMP, last_seen TIMESTAMP)\n"
    "  detection_observations(id SERIAL PK, track_id INTEGER FK→detection_tracks, "
    "timestamp TIMESTAMP, attributes JSONB)\n"
    "  For PPE: join detection_tracks to detection_classes where name='Person' (trackable). "
    "attributes has hardhat, vest, mask (true/false). "
    "A 'violation' means (attributes->>'hardhat')::boolean = false, etc. "
    "IMPORTANT: detection_observations has ONE ROW PER STATE CHANGE per person (not one per person). "
    "To count PEOPLE (unique individuals), use COUNT(DISTINCT o.track_id). "
    "COUNT(*) counts observation rows, which inflates numbers (e.g. 367 rows ≠ 367 people).\n\n"
    "Scope (reject anything else with a one-line refusal):\n"
    "• Worker/people counts\n"
    "• Hardhat compliance (counts and rates)\n"
    "• Safety vest compliance (counts and rates)\n"
    "• Overall PPE compliance\n"
    "• Brief safety summaries and recommendations\n\n"
    "Rules:\n"
    '2. Use tools ONLY if the answer to the question does not exist "on the screen."\n'
    "3. DO NOT use tools if the user asks without specifying a timeframe.\n"
    "4. Prefer numbers and percentages over prose.\n"
    "5. No greetings or filler words.\n"
    "6. Respond in 1-3 short sentences max.\n"
    "7. Never explain methodology—do not mention observation rows, queries, database, or how you arrived at the answer. State only the direct answer (e.g. '1 person' or '3 people')."
)


class LLMChat:
    """Conversational LLM backed by a VLLM-served OpenAI-compatible endpoint.

    Maintains per-session chat history so the model sees the full conversation.
    """

    def __init__(self) -> None:
        endpoint = os.environ["OPENAI_API_ENDPOINT"]
        api_key = os.environ["OPENAI_API_TOKEN"]
        model = os.getenv("OPENAI_MODEL", "llama-4-scout-17b-16e-w4a16")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

        llm = ChatOpenAI(
            base_url=endpoint,
            api_key=api_key,
            model=model,
            temperature=temperature,
            streaming=True,
        )

        tools = asyncio.run(load_tools())

        self._memory = MemorySaver()
        self._agent = create_agent(
            llm,
            tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._memory,
        )
        self._session_versions: dict[str, int] = {}

        log.info(
            "LLMChat initialised — endpoint=%s, model=%s, mcp_tools=%d",
            endpoint,
            model,
            len(tools),
        )

    def _thread_id(self, session_id: str) -> str:
        version = self._session_versions.get(session_id, 0)
        return f"{session_id}:{version}" if version else session_id

    def _build_input(self, question: str, context: str) -> dict:
        return {
            "messages": [
                SystemMessage(content=f"The user sees on the screen:\n{context}"),
                HumanMessage(content=f"User question: {question}"),
            ],
        }

    def chat(
        self,
        question: str,
        context: str,
        session_id: str = "default",
    ) -> str:
        """Send a question with context through the conversational agent.

        Every prior exchange in *session_id* is automatically included so the
        model can reference earlier questions and answers.

        Uses ainvoke because MCP tools are async-only.
        """
        log.info(
            "chat called: question=%r, session_id=%r, context_len=%d, context=%r",
            question,
            session_id,
            len(context) if context else 0,
            context,
        )

        response = asyncio.run(
            self._agent.ainvoke(
                self._build_input(question, context),
                config={"configurable": {"thread_id": self._thread_id(session_id)}},
            )
        )
        return response["messages"][-1].content

    def stream_question(
        self,
        question: str,
        context: str,
        session_id: str = "default",
    ) -> Generator[str, None, None]:
        """Stream answer tokens one chunk at a time.

        Conversation history is updated automatically once the full stream
        has been consumed.

        Uses astream because MCP tools are async-only.
        """

        async def _astream():
            chunks = []
            async for msg, _metadata in self._agent.astream(
                self._build_input(question, context),
                config={"configurable": {"thread_id": self._thread_id(session_id)}},
                stream_mode="messages",
            ):
                if (
                    isinstance(msg, AIMessageChunk)
                    and msg.content
                    and not msg.tool_calls
                ):
                    chunks.append(msg.content)
            return chunks

        for chunk in asyncio.run(_astream()):
            yield chunk

    def clear_history(self, session_id: str = "default") -> None:
        """Clear the conversation history for a session."""
        self._session_versions[session_id] = (
            self._session_versions.get(session_id, 0) + 1
        )
