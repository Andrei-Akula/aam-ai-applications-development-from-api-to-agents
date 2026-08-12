import json
from collections.abc import AsyncIterator

import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIResponsesClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Responses API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with the Responses API directly
    and handle its unique event-based streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no output text.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Uses the Responses API format with 'instructions' and 'input' parameters.
            The response is printed to stdout before being returned.
        """
        payload = self._build_payload(messages, **kwargs)
        response = requests.post(
            self._endpoint,
            headers=self._build_headers(),
            json=payload,
        )
        response.raise_for_status()

        content = self._extract_content(response.json())
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with event-based streaming.

        The Responses API uses a different SSE format than Chat Completions,
        with explicit event types and data fields.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses event-based Server-Sent Events (SSE) format.
            Listens for 'response.output_text.delta' events to build the response.
            Each line with "event: " specifies the event type, followed by "data: " with the payload.
        """
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True

        content_parts: list[str] = []
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._endpoint,
                headers=self._build_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for event_type, data in self._iter_sse_events(response):
                    if event_type == "response.output_text.delta":
                        delta = self._extract_stream_delta(data)
                        if delta:
                            print(delta, end="", flush=True)
                            content_parts.append(delta)
                    elif event_type in {"response.completed", "response.failed"}:
                        break

        print()
        return Message(role=Role.ASSISTANT, content="".join(content_parts))

    def _build_headers(self) -> dict[str, str]:
        """Build headers required by the OpenAI Responses API."""
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[Message], **kwargs) -> dict:
        """Build a Responses API request payload."""
        return {
            "model": self._model_name,
            "instructions": self._system_prompt,
            "input": [message.to_dict() for message in messages],
            **kwargs,
        }

    @staticmethod
    def _extract_content(response: dict) -> str:
        """Extract assistant text from a non-streaming Responses API response."""
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        content_parts = [
            content.get("text", "")
            for output_item in response.get("output", [])
            for content in output_item.get("content", [])
            if content.get("type") == "output_text"
        ]
        content = "".join(content_parts)
        if not content:
            raise ValueError("No output text has been present in the response")

        return content

    @staticmethod
    def _extract_stream_delta(data: str) -> str:
        """Extract text from a Responses API output-text delta event."""
        return json.loads(data).get("delta") or ""

    @staticmethod
    async def _iter_sse_events(
        response: aiohttp.ClientResponse,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield event type and data pairs from a Responses API SSE stream."""
        event_type = ""
        data_lines: list[str] = []

        async for raw_line in response.content:
            line = raw_line.decode("utf-8").rstrip("\r\n")

            if not line:
                if event_type and data_lines:
                    yield event_type, "\n".join(data_lines)
                event_type = ""
                data_lines = []
                continue

            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if event_type and data_lines:
            yield event_type, "\n".join(data_lines)
