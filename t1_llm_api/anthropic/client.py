from anthropic import Anthropic, AsyncAnthropic

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class AnthropicAIClient(AIClient):
    """
    Client for Anthropic's Claude API using the official SDK.

    This implementation uses the official Anthropic Python library to interact
    with Claude models, providing both synchronous and streaming response capabilities.

    Attributes:
        _client (Anthropic): Synchronous Anthropic client instance.
        _async_client (AsyncAnthropic): Asynchronous Anthropic client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Anthropic client with SDK.

        Args:
            endpoint (str): The Anthropic API endpoint (for compatibility, not used by SDK).
            model_name (str): The Claude model to use (e.g., 'claude-3-opus', 'claude-sonnet-4-5').
            api_key (str): The Anthropic API key for authentication.
            system_prompt (str): The system instruction to guide Claude's behavior.
        """
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
        )
        self._client = Anthropic(api_key=api_key)
        self._async_client = AsyncAnthropic(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Anthropic's Claude API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            Claude's API uses a separate 'system' parameter for system instructions.
            Response content blocks are concatenated into a single text response.
            The response is printed to stdout before being returned.
        """
        request = self._build_request(messages, **kwargs)
        response = self._client.messages.create(**request)
        content = self._extract_content(response.content)

        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Anthropic's Claude API.

        The response is streamed using event-based streaming, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Listens for 'content_block_delta' events with text deltas.
            Each delta is printed to stdout as it arrives for real-time display.
        """
        request = self._build_request(messages, **kwargs)
        content_parts = []

        async with self._async_client.messages.stream(**request) as stream:
            async for content in stream.text_stream:
                print(content, end="", flush=True)
                content_parts.append(content)

        print()
        return Message(role=Role.ASSISTANT, content="".join(content_parts))

    def _build_request(self, messages: list[Message], **kwargs) -> dict:
        """Build a request payload for Anthropic's Messages API."""
        return {
            "model": self._model_name,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "system": self._system_prompt,
            "messages": [message.to_dict() for message in messages],
            **kwargs,
        }

    @staticmethod
    def _extract_content(content_blocks) -> str:
        """Extract text from Anthropic response content blocks."""
        content = "".join(
            block.text
            for block in content_blocks
            if block.type == "text"
        )
        if not content_blocks:
            raise ValueError("No content blocks have been present in the response")

        return content
