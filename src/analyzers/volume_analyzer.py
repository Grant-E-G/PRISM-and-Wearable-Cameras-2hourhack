"""Video D analyzer: read the volume of liquid added to the bottle.

Challenge: reading graduated markings on glassware, tracking liquid levels,
handling parallax error, reading small graduations.
"""

import json
import re
from typing import Any

from src.analyzers.base_analyzer import BaseAnalyzer
from src.video_processor import extract_frames


_SYSTEM_PROMPT = """\
You are a careful lab-video analyst reading liquid volumes from glassware in a
lab video of LB Agar Plate preparation.

For EACH frame:
1. Identify any graduated glassware (bottles, cylinders, beakers, flasks).
2. Read the liquid volume indicated by the meniscus or liquid level against
   the graduation marks.
3. Note the units (mL, L, etc.) if visible.
4. Note whether liquid is being poured, added, or static.

Output **only** valid JSON in this exact format, nothing else:
{
  "frames": [
    {
      "timestamp_sec": <number>,
      "glassware_visible": <bool>,
      "volume_ml": <number or null>,
      "units": "<mL/L/etc. or null>",
      "liquid_action": "<pouring|adding|static|unclear>",
      "graduation_readable": <bool>,
      "notes": "<any observations about visibility or uncertainty>"
    }
  ],
  "volume_additions": [
    {"timestamp_sec": <number>, "volume_ml": <number>, "description": "<string>"}
  ],
  "total_volume_added_ml": <number or null>,
  "summary": "<narrative of all liquid volume events observed>"
}
"""


class VolumeAnalyzer(BaseAnalyzer):
    """Analyzer for Video D: reads liquid volumes from graduated glassware."""

    TASK_DESCRIPTION = (
        "Video D – LB Agar Plate Preparation: read the volume of liquid "
        "added to the bottle from graduated glassware markings."
    )

    def analyze(
        self,
        video_path: str,
        interval_seconds: float = 3.0,
        max_frames: int = 30,
        **kwargs: Any,
    ) -> dict:
        """Analyze the video to extract volume measurements.

        Args:
            video_path: Path to the video file.
            interval_seconds: Frame sampling interval (default 3 s).
            max_frames: Hard cap on frames to limit API cost.

        Returns:
            Dict with 'total_volume_added_ml', 'volume_additions', 'frame_details',
            'summary', 'raw_response'.
        """
        frames = extract_frames(
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )

        prompt = (
            "Analyze these frames from a lab video of LB agar plate preparation. "
            "Focus on reading liquid volumes from graduated glassware. "
            + _SYSTEM_PROMPT
        )

        raw = self.vlm.analyze_frames(frames, prompt, max_tokens=2048)
        result = _parse_json_response(raw)

        return {
            "task": "volume_reading",
            "video_path": video_path,
            "total_volume_added_ml": result.get("total_volume_added_ml"),
            "volume_additions": result.get("volume_additions", []),
            "frame_details": result.get("frames", []),
            "summary": result.get("summary", ""),
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
    return {
        "total_volume_added_ml": None,
        "volume_additions": [],
        "frames": [],
        "summary": text,
    }
