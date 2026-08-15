import json
from collections.abc import AsyncIterator

import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
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
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Anthropic's SSE format, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Listens for 'content_block_delta' events with 'text_delta' type.
            Stops processing when 'message_stop' event is received.
            Each delta is printed to stdout as it arrives.
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
                    if event_type == "content_block_delta":
                        text = self._extract_stream_text(data)
                        if text:
                            print(text, end="", flush=True)
                            content_parts.append(text)
                    elif event_type == "message_stop":
                        break

        print()
        return Message(role=Role.ASSISTANT, content="".join(content_parts))

    def _build_headers(self) -> dict[str, str]:
        """Build headers required by Anthropic's Messages API."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[Message], **kwargs) -> dict:
        """Build an Anthropic Messages API request payload."""
        return {
            "model": self._model_name,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "system": self._system_prompt,
            "messages": [message.to_dict() for message in messages],
            **kwargs,
        }

    @staticmethod
    def _extract_content(response: dict) -> str:
        """Extract assistant text from a non-streaming Anthropic response."""
        content_blocks = response.get("content", [])
        if not content_blocks:
            raise ValueError("No content blocks have been present in the response")

        return "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        )

    @staticmethod
    def _extract_stream_text(data: str) -> str:
        """Extract text from an Anthropic content-block delta event."""
        event = json.loads(data)
        delta = event.get("delta", {})
        if delta.get("type") != "text_delta":
            return ""

        return delta.get("text", "")

    @staticmethod
    async def _iter_sse_events(
        response: aiohttp.ClientResponse,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield event type and data pairs from an Anthropic SSE stream."""
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
