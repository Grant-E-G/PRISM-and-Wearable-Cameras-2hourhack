"""Video B analyzer: identify color changes during reaction intervals.

Challenge: tracking subtle or gradual appearance changes over timed intervals
(approx. every 30 minutes) in Antimony Sulfide Nanocrystal synthesis.
"""

import json
import re
from typing import Any

from src.analyzers.base_analyzer import BaseAnalyzer
from src.video_processor import extract_frames_at_timestamps, get_video_metadata


_SYSTEM_PROMPT = """\
You are a careful lab-video analyst helping extract protocol information from
a hot-injection synthesis of Antimony Sulfide Nanocrystals.

For EACH frame you receive, describe:
1. The dominant color of the reaction mixture / solution visible in the flask
   or reaction vessel.
2. Any color change compared to the previous frame.
3. A precise description of the color using standard color names or RGB/hex
   approximations where possible.

Output **only** valid JSON in this exact format, nothing else:
{
  "frames": [
    {
      "timestamp_sec": <number>,
      "color_description": "<string>",
      "change_from_previous": "<string or 'no change'>",
      "hex_approximate": "<#RRGGBB or null>"
    }
  ],
  "color_change_summary": "<narrative summary of all color transitions>",
  "notable_transitions": [
    {"from_timestamp_sec": <number>, "to_timestamp_sec": <number>,
     "description": "<string>"}
  ]
}
"""


class ColorChangeAnalyzer(BaseAnalyzer):
    """Analyzer for Video B: tracks color changes over reaction time."""

    TASK_DESCRIPTION = (
        "Video B – Antimony Sulfide Nanocrystal Synthesis: identify exact "
        "color changes during each 30-minute interval of the reaction."
    )

    def analyze(
        self,
        video_path: str,
        interval_minutes: float = 30.0,
        **kwargs: Any,
    ) -> dict:
        """Analyze color changes at regular time intervals.

        Args:
            video_path: Path to the video file.
            interval_minutes: Interval at which to sample frames (default 30 min).
                              For short demo videos the interval is auto-scaled.

        Returns:
            Dict with 'color_change_summary', 'frame_details', 'raw_response'.
        """
        meta = get_video_metadata(video_path)
        duration = meta["duration_sec"]

        interval_sec = interval_minutes * 60.0

        # Auto-scale for short demo videos so we always get ≥3 samples
        if duration > 0 and interval_sec > duration / 3:
            interval_sec = max(1.0, duration / 5)

        timestamps = []
        t = 0.0
        while t <= duration:
            timestamps.append(t)
            t += interval_sec

        frames = extract_frames_at_timestamps(video_path, timestamps)

        prompt = (
            "Analyze these frames sampled at regular intervals from a lab video "
            "of a chemical synthesis reaction. " + _SYSTEM_PROMPT
        )

        raw = self.vlm.analyze_frames(frames, prompt, max_tokens=2048)
        result = _parse_json_response(raw)

        return {
            "task": "color_change_detection",
            "video_path": video_path,
            "sampling_interval_sec": interval_sec,
            "color_change_summary": result.get("color_change_summary", ""),
            "notable_transitions": result.get("notable_transitions", []),
            "frame_details": result.get("frames", []),
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
        "color_change_summary": text,
        "frames": [],
        "notable_transitions": [],
    }
