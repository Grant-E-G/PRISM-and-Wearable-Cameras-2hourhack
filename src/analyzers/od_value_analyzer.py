"""Video C analyzer: read OD (optical density) values from each sample.

Challenge: OCR on instrument displays, small text, screen glare, reading
numeric values from spectrophotometer or plate-reader screens.
"""

import json
import re
from typing import Any

from src.analyzers.base_analyzer import BaseAnalyzer
from src.video_processor import extract_frames


_SYSTEM_PROMPT = """\
You are a careful lab-video analyst specializing in reading instrument displays
and OCR from scientific lab footage of a Yeast Transformation protocol.

For EACH frame:
1. Detect any visible instrument display or measurement readout.
2. Read all numeric values shown (OD values, wavelength, sample ID, etc.).
3. If an OD (optical density) value is visible, record it precisely.

Output **only** valid JSON in this exact format, nothing else:
{
  "frames": [
    {
      "timestamp_sec": <number>,
      "display_visible": <bool>,
      "od_readings": [
        {"sample_id": "<string or null>", "od_value": <number or null>,
         "wavelength_nm": <number or null>, "raw_text": "<string>"}
      ],
      "other_values": "<any other numbers or text visible on display>"
    }
  ],
  "all_od_values": [
    {"sample_id": "<string>", "od_value": <number>, "wavelength_nm": <number or null>}
  ],
  "notes": "<observations about display readability, glare, etc.>"
}
"""


class ODValueAnalyzer(BaseAnalyzer):
    """Analyzer for Video C: reads OD values from instrument displays."""

    TASK_DESCRIPTION = (
        "Video C – Yeast Transformation Protocol: read the OD (optical density) "
        "values from each sample as shown on instrument displays."
    )

    def analyze(
        self,
        video_path: str,
        interval_seconds: float = 2.0,
        max_frames: int = 40,
        **kwargs: Any,
    ) -> dict:
        """Analyze the video to extract all OD readings.

        Args:
            video_path: Path to the video file.
            interval_seconds: Frame sampling rate (default 2 s for dense coverage).
            max_frames: Hard cap on frames to limit API cost.

        Returns:
            Dict with 'all_od_values', 'frame_details', 'notes', 'raw_response'.
        """
        frames = extract_frames(
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )

        prompt = (
            "Analyze these frames from a lab video of a yeast transformation "
            "protocol. Focus especially on any instrument displays showing OD "
            "values or spectrophotometer readings. " + _SYSTEM_PROMPT
        )

        raw = self.vlm.analyze_frames(frames, prompt, max_tokens=2048)
        result = _parse_json_response(raw)

        return {
            "task": "od_value_reading",
            "video_path": video_path,
            "all_od_values": result.get("all_od_values", []),
            "frame_details": result.get("frames", []),
            "notes": result.get("notes", ""),
            "raw_response": raw,
        }


def _parse_json_response(text: str) -> dict:
    """Extract the first JSON object from the VLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"all_od_values": [], "frames": [], "notes": text}
