import json
from collections.abc import AsyncIterator

import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.openai.base import BaseOpenAIClient


class CustomOpenAIClient(BaseOpenAIClient):
    """
    Custom HTTP client for OpenAI Chat Completions API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, providing more control over the HTTP layer and demonstrating
    how to interact with the API directly.
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
            ValueError: If the API response contains no choices.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The system prompt is automatically prepended to the messages.
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

        The response is streamed token-by-token using OpenAI's SSE format,
        with each chunk printed immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters for the API (currently unused).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The system prompt is automatically prepended to the messages.
            Each token is printed to stdout as it arrives.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
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
                async for data in self._iter_sse_data(response):
                    if data == "[DONE]":
                        break

                    content = self._extract_stream_content(json.loads(data))
                    if content:
                        print(content, end="", flush=True)
                        content_parts.append(content)

        print()
        return Message(role=Role.ASSISTANT, content="".join(content_parts))

    def _build_headers(self) -> dict[str, str]:
        """Build headers required by the OpenAI Chat Completions API."""
        return {
            "Authorization": self._api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[Message], **kwargs) -> dict:
        """Build a Chat Completions request payload."""
        request_messages = [
            Message(role=Role.SYSTEM, content=self._system_prompt),
            *messages,
        ]
        return {
            "model": self._model_name,
            "messages": [message.to_dict() for message in request_messages],
            **kwargs,
        }

    @staticmethod
    def _extract_content(completion: dict) -> str:
        """Extract assistant text from a non-streaming completion."""
        choices = completion.get("choices", [])
        if not choices:
            raise ValueError("No choices have been present in the response")

        return choices[0].get("message", {}).get("content") or ""

    @staticmethod
    def _extract_stream_content(chunk: dict) -> str:
        """Extract text from a streaming Chat Completions chunk."""
        choices = chunk.get("choices", [])
        if not choices:
            return ""

        return choices[0].get("delta", {}).get("content") or ""

    @staticmethod
    async def _iter_sse_data(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
        """Yield data payloads from an OpenAI Server-Sent Events response."""
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if line.startswith("data:"):
                yield line.removeprefix("data:").strip()
