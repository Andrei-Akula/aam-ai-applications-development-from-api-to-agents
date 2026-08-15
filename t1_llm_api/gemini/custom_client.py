import json
from collections.abc import AsyncIterator

import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
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
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        response = requests.post(
            self._build_url("generateContent"),
            headers=self._build_headers(),
            json=self._build_payload(messages, **kwargs),
        )
        response.raise_for_status()

        content = self._extract_content(response.json())
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        payload = self._build_payload(messages, **kwargs)
        content_parts: list[str] = []

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._build_url("streamGenerateContent?alt=sse"),
                headers=self._build_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for data in self._iter_sse_data(response):
                    content = self._extract_stream_content(data)
                    if content:
                        print(content, end="", flush=True)
                        content_parts.append(content)

        print()
        return Message(role=Role.ASSISTANT, content="".join(content_parts))

    def _build_url(self, operation: str) -> str:
        """Build a Gemini model operation URL."""
        return f"{self._endpoint}/{self._model_name}:{operation}"

    def _build_headers(self) -> dict[str, str]:
        """Build headers required by Gemini's generate-content API."""
        return {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[Message], **kwargs) -> dict:
        """Build a Gemini generate-content request payload."""
        request_kwargs = dict(kwargs)
        generation_config = dict(request_kwargs.pop("generationConfig", {}))
        max_tokens = request_kwargs.pop(
            "max_tokens",
            generation_config.pop("max_tokens", 1024),
        )
        generation_config.setdefault(
            "maxOutputTokens",
            generation_config.pop("max_output_tokens", max_tokens),
        )

        return {
            "system_instruction": {
                "parts": [{"text": self._system_prompt}],
            },
            "contents": self._build_contents(messages),
            "generationConfig": generation_config,
            **request_kwargs,
        }

    @staticmethod
    def _build_contents(messages: list[Message]) -> list[dict]:
        """Convert shared messages to Gemini content objects."""
        contents = []
        for message in messages:
            if message.role == Role.SYSTEM:
                continue

            role = "model" if message.role == Role.ASSISTANT else "user"
            contents.append({
                "role": role,
                "parts": [{"text": message.content}],
            })

        return contents

    @staticmethod
    def _extract_content(response: dict) -> str:
        """Extract text from a non-streaming Gemini response."""
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates have been present in the response")

        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    @staticmethod
    def _extract_stream_content(data: str) -> str:
        """Extract text from a Gemini streaming response chunk."""
        return CustomGeminiAIClient._extract_content(json.loads(data))

    @staticmethod
    async def _iter_sse_data(
        response: aiohttp.ClientResponse,
    ) -> AsyncIterator[str]:
        """Yield data payloads from a Gemini Server-Sent Events stream."""
        data_lines: list[str] = []

        async for raw_line in response.content:
            line = raw_line.decode("utf-8").rstrip("\r\n")

            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                data_lines = []
                continue

            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if data_lines:
            yield "\n".join(data_lines)
