"""Minimal yeast transformation protocol demo analyzer."""

import json
import re
from typing import Any

from src.analyzers.base_analyzer import BaseAnalyzer
from src.video_processor import extract_frames


_PROMPT = """\
You are analyzing wearable-camera footage of a yeast transformation protocol.

Extract only information supported by the frames. Focus on:
- OD values visible on instruments or labels
- Reagents, sample identifiers, temperatures, timings, volumes, and equipment
- Human actions that matter for reproducing the protocol
- Tacit details such as mixing, incubation setup, tube handling, or plating

Output only valid JSON in this exact shape:
{
  "od_values": [
    {"sample_id": "<string>", "od_value": <number>, "wavelength_nm": <number or null>}
  ],
  "observed_actions": [
    {
      "timestamp_sec": <number>,
      "action": "<short action>",
      "materials": ["<material or equipment>", "..."],
      "measurement": "<visible value or null>",
      "confidence": "<high|medium|low>"
    }
  ],
  "protocol": {
    "title": "Yeast Transformation Protocol",
    "materials": ["<visible or strongly implied material>", "..."],
    "steps": ["<chronological protocol step>", "..."],
    "uncertainties": ["<missing value or inferred-only detail>", "..."]
  },
  "notes": "<brief notes on visibility and limitations>"
}
"""


def analyze_yeast_transformation(
    vlm: Any,
    video_path: str,
    interval_seconds: float = 2.0,
    max_frames: int = 40,
) -> dict:
    """Return a compact yeast transformation protocol demo result."""
    frames = extract_frames(
        video_path,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )
    raw = vlm.analyze_frames(frames, _PROMPT, max_tokens=3072)
    return _result(video_path, raw, _parse_json_response(raw))


class YeastTransformationAnalyzer(BaseAnalyzer):
    """Analyzer for Video C: yeast transformation protocol capture."""

    TASK_DESCRIPTION = (
        "Video C - Yeast Transformation Protocol: extract OD readings, "
        "observed protocol actions, and a compact protocol draft."
    )

    def analyze(
        self,
        video_path: str,
        interval_seconds: float = 2.0,
        max_frames: int = 40,
        **kwargs: Any,
    ) -> dict:
        return analyze_yeast_transformation(
            self.vlm,
            video_path,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )


def _result(video_path: str, raw: str, data: dict) -> dict:
    protocol = data.get("protocol") or {}
    return {
        "task": "yeast_transformation_protocol",
        "video_path": video_path,
        "od_values": data.get("od_values", []),
        "observed_actions": data.get("observed_actions", []),
        "protocol": {
            "title": protocol.get("title", "Yeast Transformation Protocol"),
            "materials": protocol.get("materials", []),
            "steps": protocol.get("steps", []),
            "uncertainties": protocol.get("uncertainties", []),
        },
        "notes": data.get("notes", ""),
        "raw_response": raw,
    }


def _parse_json_response(text: str) -> dict:
    """Extract the first JSON object from a VLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "od_values": [],
        "observed_actions": [],
        "protocol": {
            "title": "Yeast Transformation Protocol",
            "materials": [],
            "steps": [],
            "uncertainties": ["Could not parse structured JSON from model output."],
        },
        "notes": text,
    }
