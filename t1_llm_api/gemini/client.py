from google import genai
from google.genai import types

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class GeminiAIClient(AIClient):
    """
    Client for Google Gemini API using the official SDK.

    This implementation uses the official Google GenAI Python library to interact
    with Gemini models, providing both synchronous and streaming response capabilities.

    Attributes:
        _client (genai.Client): Google GenAI client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Gemini client with SDK.

        Args:
            endpoint (str): The Gemini API endpoint (for compatibility, not used by SDK).
            model_name (str): The Gemini model to use (e.g., 'gemini-3-flash-preview').
            api_key (str): The Google API key for authentication.
            system_prompt (str): The system instruction to guide the model's behavior.
        """
        super().__init__(
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
        )
        self._client = genai.Client(api_key=api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Google's Gemini API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            Gemini uses 'system_instruction' parameter for system-level guidance.
            The response is printed to stdout before being returned.
        """
        request = self._build_request(messages, **kwargs)
        response = self._client.models.generate_content(**request)
        content = response.text or ""

        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Google's Gemini API.

        The response is streamed chunk-by-chunk, with each text chunk printed
        immediately as it arrives.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            Uses the async streaming interface provided by the Gemini SDK.
            Each chunk's text is printed to stdout as it arrives.
        """
        request = self._build_request(messages, **kwargs)
        content_parts: list[str] = []

        stream = await self._client.aio.models.generate_content_stream(**request)
        async for chunk in stream:
            content = chunk.text or ""
            if content:
                print(content, end="", flush=True)
                content_parts.append(content)

        print()
        return Message(role=Role.ASSISTANT, content="".join(content_parts))

    def _build_request(self, messages: list[Message], **kwargs) -> dict:
        """Build a request for Gemini's generate-content API."""
        return {
            "model": self._model_name,
            "contents": self._build_contents(messages),
            "config": self._build_config(**kwargs),
        }

    def _build_config(self, **kwargs) -> types.GenerateContentConfig:
        """Build Gemini generation configuration with the system prompt."""
        config_kwargs = dict(kwargs)
        max_tokens = config_kwargs.pop("max_tokens", 1024)
        config_kwargs.setdefault("max_output_tokens", max_tokens)
        config_kwargs["system_instruction"] = self._system_prompt
        return types.GenerateContentConfig(**config_kwargs)

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
