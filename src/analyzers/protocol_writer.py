"""Stretch goal: generate a structured lab protocol from video + extracted data.

Combines VLM narration of the video with previously extracted structured
information (well counts, OD values, volumes, color changes) to produce a
complete, reproducible text protocol.
"""

import json
from typing import Any

from src.analyzers.base_analyzer import BaseAnalyzer
from src.video_processor import extract_frames


_PROTOCOL_PROMPT_TEMPLATE = """\
You are an expert scientific protocol writer working with PRISM
(Protocol Recording for Improved Scientific Methodology).

You have been given:
1. A set of video frames from a lab procedure.
2. Structured data extracted from those frames (below).

Your task: write a complete, reproducible step-by-step scientific protocol in
the style of a methods section.  Include:
- All materials and reagents you can identify (with quantities where available)
- Numbered procedural steps in chronological order
- Specific values extracted from the video (OD readings, volumes, color
  changes, well counts, timing, etc.)
- Notes on critical steps, safety, or tacit knowledge visible in the video

Extracted data:
{extracted_data}

Write the protocol now, starting with a title, then Materials, then Procedure.
Use precise scientific language suitable for publication.
"""


class ProtocolWriter(BaseAnalyzer):
    """Stretch goal: synthesise a full written protocol from video + data."""

    TASK_DESCRIPTION = (
        "Stretch Goal – Protocol Writer: generate a complete step-by-step "
        "scientific protocol from the video and all extracted information."
    )

    def analyze(
        self,
        video_path: str,
        extracted_results: dict | None = None,
        interval_seconds: float = 10.0,
        max_frames: int = 20,
        **kwargs: Any,
    ) -> dict:
        """Generate a written protocol for the lab video.

        Args:
            video_path: Path to the video file.
            extracted_results: Dict of previously extracted structured data
                               (e.g. from WellPlateAnalyzer, ODValueAnalyzer,
                               VolumeAnalyzer, ColorChangeAnalyzer).
            interval_seconds: Frame sampling interval for narration context.
            max_frames: Hard cap on frames sent to the VLM.

        Returns:
            Dict with 'protocol_text' and 'raw_response'.
        """
        frames = extract_frames(
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )

        # Serialise any previously extracted data for injection into the prompt
        if extracted_results:
            extracted_str = json.dumps(extracted_results, indent=2)
        else:
            extracted_str = "(No pre-extracted data provided — infer from frames.)"

        prompt = _PROTOCOL_PROMPT_TEMPLATE.format(extracted_data=extracted_str)

        raw = self.vlm.analyze_frames(frames, prompt, max_tokens=4096)

        return {
            "task": "protocol_writing",
            "video_path": video_path,
            "protocol_text": raw,
            "raw_response": raw,
        }
