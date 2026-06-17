"""VLM client supporting both Anthropic (Claude) and Google (Gemini) APIs."""

import base64
import os
from enum import Enum
from typing import Any


class VLMProvider(str, Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"


class VLMClient:
    """Thin wrapper around Claude and Gemini APIs for vision + text tasks.

    Reads API keys from environment variables:
      - ANTHROPIC_API_KEY  for Claude
      - GOOGLE_API_KEY     for Gemini
    """

    # Default model identifiers
    DEFAULT_CLAUDE_MODEL = "claude-opus-4-5"
    DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"

    def __init__(
        self,
        provider: VLMProvider | str = VLMProvider.CLAUDE,
        model: str | None = None,
    ) -> None:
        self.provider = VLMProvider(provider)
        self._client: Any = None

        if self.provider == VLMProvider.CLAUDE:
            self.model = model or self.DEFAULT_CLAUDE_MODEL
            self._init_claude()
        else:
            self.model = model or self.DEFAULT_GEMINI_MODEL
            self._init_gemini()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_claude(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(api_key=api_key)

    def _init_gemini(self) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is not set."
            )
        import google.generativeai as genai  # noqa: PLC0415

        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(self.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frames(
        self,
        frames: list[dict],
        prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Send one or more video frames plus a text prompt to the VLM.

        Args:
            frames: List of frame dicts from video_processor (must contain
                    'image_b64' and 'timestamp_sec').
            prompt: Instruction / question for the model.
            max_tokens: Maximum tokens in the response.

        Returns:
            Model response as a plain string.
        """
        if self.provider == VLMProvider.CLAUDE:
            return self._analyze_claude(frames, prompt, max_tokens)
        return self._analyze_gemini(frames, prompt, max_tokens)

    def analyze_image(
        self,
        image_b64: str,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Analyze a single base64-encoded JPEG image with a text prompt.

        Args:
            image_b64: Base64-encoded JPEG.
            prompt: Instruction / question.
            max_tokens: Maximum tokens in the response.

        Returns:
            Model response as a plain string.
        """
        frame = {"image_b64": image_b64, "timestamp_sec": 0}
        return self.analyze_frames([frame], prompt, max_tokens)

    # ------------------------------------------------------------------
    # Provider-specific helpers
    # ------------------------------------------------------------------

    def _analyze_claude(
        self, frames: list[dict], prompt: str, max_tokens: int
    ) -> str:
        """Build a multi-image Claude message and return the text response."""
        content: list[dict] = []

        for frame in frames:
            ts = frame.get("timestamp_sec", "?")
            content.append(
                {
                    "type": "text",
                    "text": f"[Frame at {ts}s]",
                }
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame["image_b64"],
                    },
                }
            )

        content.append({"type": "text", "text": prompt})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def _analyze_gemini(
        self, frames: list[dict], prompt: str, max_tokens: int
    ) -> str:
        """Build a Gemini multi-image message and return the text response."""
        import google.generativeai as genai  # noqa: PLC0415

        parts = []
        for frame in frames:
            ts = frame.get("timestamp_sec", "?")
            parts.append(f"[Frame at {ts}s]")
            image_data = base64.b64decode(frame["image_b64"])
            parts.append(
                genai.types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
            )

        parts.append(prompt)

        response = self._client.generate_content(
            parts,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens
            ),
        )
        return response.text
