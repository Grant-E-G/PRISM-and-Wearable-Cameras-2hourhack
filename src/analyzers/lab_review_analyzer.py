"""Budgeted Claude review of lab video with sparse feedback sampling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.analyzers.base_analyzer import BaseAnalyzer
from src.video_processor import (
    extract_frames_at_global_timestamps,
    extract_frames_from_video_set,
    get_video_set_metadata,
)


FIRST_PASS_PROMPT = """\
You are reviewing sparse wearable-camera frames from a laboratory video.

Goal of this pass:
- Figure out what happened and approximately when.
- Identify low-information stretches and avoid over-interpreting them.
- Request only the before/after shots that would materially improve the review.

Return only valid JSON in this shape:
{
  "video_summary": "<brief chronological summary>",
  "event_timeline": [
    {
      "timestamp_sec": <number>,
      "event": "<what appears to be happening>",
      "evidence": "<visible evidence>",
      "confidence": "<high|medium|low>"
    }
  ],
  "focus_requests": [
    {
      "start_sec": <number>,
      "end_sec": <number>,
      "reason": "<why denser before/after shots are needed>",
      "priority": "<high|medium|low>"
    }
  ],
  "low_information_ranges": [
    {"start_sec": <number>, "end_sec": <number>, "reason": "<why low information>"}
  ],
  "notes": "<visibility limits and uncertainty>"
}

Keep focus_requests sparse. Prefer short windows around visible actions, setup
changes, measurements, labeling, transfers, incubations, instrument displays,
and any step where reproducibility would depend on tacit details.
"""


DETAIL_PASS_PROMPT = """\
You are reviewing a denser set of frames from one requested lab-video window.

Window requested:
{window_json}

Use only visible evidence. Extract concrete lab actions and reproducibility
advice. If more context would change the review, request a small adjacent
window, but do not request broad scanning.

Return only valid JSON in this shape:
{
  "observed_actions": [
    {
      "timestamp_sec": <number>,
      "action": "<short action>",
      "materials": ["<material or equipment>", "..."],
      "measurement": "<visible value or null>",
      "confidence": "<high|medium|low>"
    }
  ],
  "reproducibility_risks": [
    {
      "timestamp_sec": <number>,
      "action": "<observed action or omission>",
      "issue": "<why this could make the work hard to reproduce>",
      "severity": "<Very High|High|Medium|Low>",
      "suggested_fix": "<concrete correction or note to capture>",
      "confidence": "<high|medium|low>"
    }
  ],
  "thumbs_up": [
    {
      "timestamp_sec": <number>,
      "practice": "<good practice observed>",
      "why_it_helps": "<how this improves reproducibility>",
      "confidence": "<high|medium|low>"
    }
  ],
  "focus_requests": [
    {
      "start_sec": <number>,
      "end_sec": <number>,
      "reason": "<small adjacent context still needed>",
      "priority": "<high|medium|low>"
    }
  ],
  "notes": "<visibility limits and uncertainty>"
}
"""


FINAL_SYNTHESIS_PROMPT = """\
You are producing the final review from a budgeted, sparse lab-video analysis.
The input contains a coarse first pass plus any denser follow-up windows that
fit the request budget.

Return only valid JSON in this shape:
{
  "review_summary": "<what happened and when, with uncertainty>",
  "event_timeline": [
    {
      "timestamp_sec": <number>,
      "event": "<chronological event>",
      "evidence": "<visible evidence>",
      "confidence": "<high|medium|low>"
    }
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
  "reproducibility_risks": [
    {
      "timestamp_sec": <number>,
      "action": "<observed action or omission>",
      "issue": "<why this could make the work hard to reproduce>",
      "severity": "<Very High|High|Medium|Low>",
      "suggested_fix": "<concrete correction or note to capture>",
      "confidence": "<high|medium|low>"
    }
  ],
  "thumbs_up": [
    {
      "timestamp_sec": <number>,
      "practice": "<good practice observed>",
      "why_it_helps": "<how this improves reproducibility>",
      "confidence": "<high|medium|low>"
    }
  ],
  "reproducibility_metrics": [
    {
      "metric": "<critical-parameters|materials-identification|action-order|measurement-capture|timing-capture>",
      "score": <integer 0-5>,
      "evidence": "<why this score is justified>",
      "recommendation": "<how to improve capture or protocol>"
    }
  ],
  "meta_advice": {
    "overall_review": "<plain-language review of how well the procedure was captured>",
    "next_time": [
      "<concrete change that would make the next recording/protocol more reproducible>",
      "..."
    ]
  },
  "protocol": {
    "title": "Lab Video Review",
    "materials": ["<visible or strongly implied material>", "..."],
    "steps": ["<chronological protocol step>", "..."],
    "uncertainties": ["<missing value or inferred-only detail>", "..."]
  },
  "notes": "<important limitations>"
}

Analysis input:
{analysis_json}
"""


@dataclass
class RequestBudget:
    """Tracks a hard cap on Claude requests made by this analyzer."""

    maximum: int
    used: int = 0

    def spend(self) -> None:
        if self.used >= self.maximum:
            raise RuntimeError("Claude request budget exhausted.")
        self.used += 1

    @property
    def remaining(self) -> int:
        return max(0, self.maximum - self.used)

    def as_dict(self) -> dict:
        return {
            "max_claude_requests": self.maximum,
            "used_claude_requests": self.used,
            "remaining_claude_requests": self.remaining,
        }


class LabReviewAnalyzer(BaseAnalyzer):
    """Claude-only sparse review for long, mostly low-information lab videos."""

    TASK_DESCRIPTION = (
        "Budgeted lab review: sparse first pass, Claude-requested detail "
        "windows, and reproducibility advice."
    )

    def analyze(
        self,
        video_path: str,
        coarse_interval_seconds: float = 30.0,
        coarse_max_frames: int = 18,
        detail_interval_seconds: float = 3.0,
        frames_per_focus: int = 8,
        max_focus_windows: int = 6,
        max_claude_requests: int = 6,
        max_sampled_frames: int = 70,
        **kwargs: Any,
    ) -> dict:
        """Run a sparse, budgeted Claude review and return viewer annotations."""
        _require_claude(self.vlm)
        if max_claude_requests < 1:
            raise ValueError("max_claude_requests must be at least 1.")
        if max_sampled_frames < 1:
            raise ValueError("max_sampled_frames must be at least 1.")

        budget = RequestBudget(maximum=max_claude_requests)
        metadata = get_video_set_metadata(video_path)

        coarse_limit = min(coarse_max_frames, max_sampled_frames)
        coarse_frames = extract_frames_from_video_set(
            video_path,
            interval_seconds=coarse_interval_seconds,
            max_frames=coarse_limit,
        )
        budget.spend()
        first_raw = self.vlm.analyze_frames(
            coarse_frames,
            FIRST_PASS_PROMPT,
            max_tokens=3072,
        )
        first_pass = _parse_json_response(first_raw, default=_default_first_pass(first_raw))

        sampled_frames = len(coarse_frames)
        focus_queue = _normalize_focus_requests(
            first_pass.get("focus_requests", []),
            duration_sec=metadata.get("duration_sec", 0),
        )

        detail_passes: list[dict] = []
        seen_windows: set[tuple[float, float]] = set()

        while (
            focus_queue
            and len(detail_passes) < max_focus_windows
            and budget.remaining > 1
            and sampled_frames < max_sampled_frames
        ):
            window = focus_queue.pop(0)
            key = (round(window["start_sec"], 1), round(window["end_sec"], 1))
            if key in seen_windows:
                continue
            seen_windows.add(key)

            remaining_frame_budget = max_sampled_frames - sampled_frames
            frame_limit = min(frames_per_focus, remaining_frame_budget)
            timestamps = _timestamps_for_window(
                window["start_sec"],
                window["end_sec"],
                detail_interval_seconds,
                frame_limit,
            )
            if not timestamps:
                continue

            frames = extract_frames_at_global_timestamps(video_path, timestamps)
            if not frames:
                continue

            sampled_frames += len(frames)
            budget.spend()
            detail_raw = self.vlm.analyze_frames(
                frames,
                DETAIL_PASS_PROMPT.replace(
                    "{window_json}", json.dumps(window, indent=2, sort_keys=True)
                ),
                max_tokens=3072,
            )
            detail_data = _parse_json_response(
                detail_raw,
                default=_default_detail_pass(detail_raw),
            )
            detail_passes.append(
                {
                    "requested_window": window,
                    "sampled_timestamps_sec": [frame["timestamp_sec"] for frame in frames],
                    "analysis": detail_data,
                    "raw_response": detail_raw,
                }
            )

            if len(detail_passes) < max_focus_windows and budget.remaining > 1:
                followups = _normalize_focus_requests(
                    detail_data.get("focus_requests", []),
                    duration_sec=metadata.get("duration_sec", 0),
                )
                focus_queue.extend(followups)

        synthesis_input = {
            "video_metadata": metadata,
            "first_pass": first_pass,
            "detail_passes": [
                {
                    "requested_window": item["requested_window"],
                    "sampled_timestamps_sec": item["sampled_timestamps_sec"],
                    "analysis": item["analysis"],
                }
                for item in detail_passes
            ],
        }

        final_raw = ""
        if budget.remaining > 0:
            budget.spend()
            final_raw = self.vlm.analyze_frames(
                [],
                FINAL_SYNTHESIS_PROMPT.replace(
                    "{analysis_json}",
                    json.dumps(
                        synthesis_input,
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                max_tokens=4096,
            )
            final = _parse_json_response(final_raw, default={})
        else:
            final = {}

        return _build_result(
            video_path=video_path,
            metadata=metadata,
            first_pass=first_pass,
            detail_passes=detail_passes,
            final=final,
            final_raw=final_raw,
            budget=budget,
            sampled_frames=sampled_frames,
            sampling_strategy={
                "coarse_interval_seconds": coarse_interval_seconds,
                "coarse_max_frames": coarse_max_frames,
                "detail_interval_seconds": detail_interval_seconds,
                "frames_per_focus": frames_per_focus,
                "max_focus_windows": max_focus_windows,
                "max_sampled_frames": max_sampled_frames,
                "coarse_sampled_timestamps_sec": [
                    frame["timestamp_sec"] for frame in coarse_frames
                ],
                "video_source_type": "folder"
                if metadata.get("is_video_set")
                else "file",
            },
            unfilled_focus_requests=focus_queue,
        )


def _require_claude(vlm: Any) -> None:
    provider = getattr(vlm, "provider", "claude")
    provider_value = getattr(provider, "value", provider)
    if provider_value != "claude":
        raise ValueError("lab_review is Claude-only. Run with --provider claude.")


def _parse_json_response(text: str, default: dict | None = None) -> dict:
    """Extract the first JSON object from a model response."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return default or {}


def _default_first_pass(raw: str) -> dict:
    return {
        "video_summary": "",
        "event_timeline": [],
        "focus_requests": [],
        "low_information_ranges": [],
        "notes": raw,
    }


def _default_detail_pass(raw: str) -> dict:
    return {
        "observed_actions": [],
        "reproducibility_risks": [],
        "thumbs_up": [],
        "focus_requests": [],
        "notes": raw,
    }


def _normalize_focus_requests(items: Any, duration_sec: float) -> list[dict]:
    if not isinstance(items, list):
        return []

    normalized = []
    duration = max(0.0, float(duration_sec or 0))
    for item in items:
        if not isinstance(item, dict):
            continue
        start = _to_float(item.get("start_sec"))
        end = _to_float(item.get("end_sec"))
        if start is None or end is None:
            continue
        start = max(0.0, start)
        end = max(start, end)
        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        if end <= start:
            continue
        normalized.append(
            {
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "reason": str(item.get("reason") or "More context requested."),
                "priority": str(item.get("priority") or "medium"),
            }
        )

    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        normalized,
        key=lambda request: (
            priority_order.get(request["priority"].lower(), 1),
            request["start_sec"],
        ),
    )


def _timestamps_for_window(
    start_sec: float,
    end_sec: float,
    interval_seconds: float,
    max_frames: int,
) -> list[float]:
    if max_frames <= 0:
        return []
    interval = max(0.25, float(interval_seconds or 1.0))
    timestamps = []
    current = start_sec
    while current <= end_sec and len(timestamps) < max_frames:
        timestamps.append(round(current, 2))
        current += interval
    if timestamps and timestamps[-1] < end_sec and len(timestamps) < max_frames:
        timestamps.append(round(end_sec, 2))
    if not timestamps:
        timestamps.append(round(start_sec, 2))
    return timestamps[:max_frames]


def _build_result(
    video_path: str,
    metadata: dict,
    first_pass: dict,
    detail_passes: list[dict],
    final: dict,
    final_raw: str,
    budget: RequestBudget,
    sampled_frames: int,
    sampling_strategy: dict,
    unfilled_focus_requests: list[dict],
) -> dict:
    merged_actions = _merge_detail_items(detail_passes, "observed_actions")
    merged_risks = _merge_detail_items(detail_passes, "reproducibility_risks")
    merged_good = _merge_detail_items(detail_passes, "thumbs_up")

    protocol = final.get("protocol") or {}
    return {
        "task": "lab_review",
        "video_path": video_path,
        "video_metadata": metadata,
        "review_summary": final.get("review_summary")
        or first_pass.get("video_summary", ""),
        "event_timeline": final.get("event_timeline")
        or first_pass.get("event_timeline", []),
        "observed_actions": final.get("observed_actions") or merged_actions,
        "reproducibility_risks": final.get("reproducibility_risks") or merged_risks,
        "thumbs_up": final.get("thumbs_up") or merged_good,
        "reproducibility_metrics": final.get("reproducibility_metrics", []),
        "meta_advice": final.get("meta_advice", {}),
        "protocol": {
            "title": protocol.get("title", "Lab Video Review"),
            "materials": protocol.get("materials", []),
            "steps": protocol.get("steps", []),
            "uncertainties": protocol.get("uncertainties", []),
        },
        "sampling_strategy": sampling_strategy,
        "request_budget": {
            **budget.as_dict(),
            "sampled_frames": sampled_frames,
            "unfilled_focus_requests": unfilled_focus_requests,
        },
        "first_pass": first_pass,
        "detail_passes": detail_passes,
        "notes": final.get("notes") or first_pass.get("notes", ""),
        "raw_response": final_raw,
    }


def _merge_detail_items(detail_passes: list[dict], key: str) -> list[dict]:
    items = []
    for detail in detail_passes:
        analysis = detail.get("analysis") or {}
        values = analysis.get(key) or []
        if isinstance(values, list):
            items.extend(values)
    return sorted(items, key=lambda item: _to_float(item.get("timestamp_sec")) or 0)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
