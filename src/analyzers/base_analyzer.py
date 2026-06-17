"""Base class for all PRISM lab video analyzers."""

from abc import ABC, abstractmethod
from typing import Any

from src.vlm_client import VLMClient
from src.video_processor import extract_frames, get_video_metadata


class BaseAnalyzer(ABC):
    """Abstract base for task-specific VLM analyzers.

    Subclasses implement ``analyze`` and provide a ``TASK_DESCRIPTION``.
    """

    TASK_DESCRIPTION: str = "Generic lab video analyzer"

    def __init__(self, vlm_client: VLMClient) -> None:
        self.vlm = vlm_client

    @abstractmethod
    def analyze(self, video_path: str, **kwargs: Any) -> dict:
        """Run the analysis on the video and return structured results.

        Args:
            video_path: Path to the lab video file.
            **kwargs: Additional task-specific parameters.

        Returns:
            Dict containing extracted protocol information.
        """

    def _extract_frames(
        self,
        video_path: str,
        interval_seconds: float = 5.0,
        max_frames: int | None = None,
    ) -> list[dict]:
        """Convenience wrapper for frame extraction."""
        return extract_frames(
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )

    def _get_metadata(self, video_path: str) -> dict:
        """Return video metadata."""
        return get_video_metadata(video_path)
