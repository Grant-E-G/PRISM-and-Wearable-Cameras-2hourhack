"""Video A analyzer: count the number of wells pipetted into in a 96-well plate.

Challenge: object interactions (pipette tip entering well), object continuity,
counting individual wells across frames.
"""

import json
import re
from typing import Any

from src.analyzers.base_analyzer import BaseAnalyzer
from src.video_processor import extract_frames


_SYSTEM_PROMPT = """\
You are a careful lab-video analyst helping extract protocol information from
a Dictyostelium growth assay video in which a scientist pipettes samples into
a 96-well plate.

For EACH frame you receive, answer the following:
1. Is a pipetting action visible? (yes/no)
2. Which wells have been pipetted into so far in this frame? Count them and, if
   visible, list their row/column positions (e.g. A1, B3, …).
3. What is your cumulative running total of wells pipetted into up to this frame?

Output **only** valid JSON in this exact format, nothing else:
{
  "frames": [
    {
      "timestamp_sec": <number>,
      "pipetting_visible": <bool>,
      "wells_identified": [<string>, ...],
      "running_total": <int>
    }
  ],
  "final_well_count": <int>,
  "notes": "<any additional observations>"
}
"""


class WellPlateAnalyzer(BaseAnalyzer):
    """Analyzer for Video A: counts wells pipetted in a 96-well plate."""

    TASK_DESCRIPTION = (
        "Video A – Dictyostelium Growth Assay: identify the number of wells "
        "in the 96-well plate the scientist pipetted into."
    )

    def analyze(
        self,
        video_path: str,
        interval_seconds: float = 3.0,
        max_frames: int = 30,
        **kwargs: Any,
    ) -> dict:
        """Analyze the video and return well-count information.

        Args:
            video_path: Path to the video file.
            interval_seconds: How often to sample frames (default: every 3 s).
            max_frames: Hard cap on frames sent to the VLM.

        Returns:
            Dict with 'final_well_count', 'frame_details', and 'raw_response'.
        """
        frames = extract_frames(
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )

        prompt = (
            "Analyze these sequential frames from a lab video of a 96-well "
            "plate pipetting experiment. " + _SYSTEM_PROMPT
        )

        raw = self.vlm.analyze_frames(frames, prompt, max_tokens=2048)
        result = _parse_json_response(raw)

        return {
            "task": "well_plate_counting",
            "video_path": video_path,
            "final_well_count": result.get("final_well_count"),
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
    return {"final_well_count": None, "frames": [], "notes": text}
