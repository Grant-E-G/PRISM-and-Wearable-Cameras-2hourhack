"""Budgeted Claude review of lab video with sparse feedback sampling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
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

Budget context:
{budget_json}
"""


DETAIL_PASS_PROMPT = """\
You are reviewing a denser set of frames from one requested lab-video window.

Window requested:
{window_json}

Budget context:
{budget_json}

Use only visible evidence. Extract concrete lab actions and reproducibility
advice. If more context would change the review, request a small adjacent
window, but do not request broad scanning. Do not request windows that overlap
or mostly duplicate already-reviewed windows. If only one or two follow-up
windows remain in the budget, request only the most reproducibility-critical
moments.

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

Keep this final response compact. Do not repeat every observed action/risk from
the detail passes; those are already stored separately. Focus on summary,
metrics, meta advice, protocol outline, and the most important timeline events.

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
        checkpoint_path: str | None = None,
        resume_checkpoint: str | None = None,
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
        sampling_strategy = {
            "coarse_interval_seconds": coarse_interval_seconds,
            "coarse_max_frames": coarse_max_frames,
            "detail_interval_seconds": detail_interval_seconds,
            "frames_per_focus": frames_per_focus,
            "max_focus_windows": max_focus_windows,
            "max_sampled_frames": max_sampled_frames,
            "video_source_type": "folder"
            if metadata.get("is_video_set")
            else "file",
        }

        checkpoint = _load_checkpoint(resume_checkpoint) if resume_checkpoint else {}
        if checkpoint:
            metadata = checkpoint.get("metadata") or metadata
            first_pass = checkpoint.get("first_pass") or _default_first_pass("")
            detail_passes = checkpoint.get("detail_passes") or []
            focus_queue = checkpoint.get("focus_queue") or []
            sampled_frames = int(checkpoint.get("sampled_frames") or 0)
            budget.used = int((checkpoint.get("request_budget") or {}).get("used_claude_requests") or 0)
            seen_windows = {
                tuple(item)
                for item in checkpoint.get("seen_windows", [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            }
            sampling_strategy.update(checkpoint.get("sampling_strategy") or {})
        else:
            first_pass = {}
            detail_passes: list[dict] = []
            focus_queue = []
            sampled_frames = 0
            seen_windows: set[tuple[float, float]] = set()

        if not first_pass:
            coarse_limit = min(coarse_max_frames, max_sampled_frames)
            coarse_frames = extract_frames_from_video_set(
                video_path,
                interval_seconds=coarse_interval_seconds,
                max_frames=coarse_limit,
            )
            budget.spend()
            try:
                first_raw = self.vlm.analyze_frames(
                    coarse_frames,
                    FIRST_PASS_PROMPT.replace(
                        "{budget_json}",
                        json.dumps(
                            _budget_context(
                                budget=budget,
                                sampled_frames=0,
                                max_sampled_frames=max_sampled_frames,
                                detail_passes=detail_passes,
                                max_focus_windows=max_focus_windows,
                                frames_per_focus=frames_per_focus,
                                reviewed_windows=[],
                            ),
                            indent=2,
                            sort_keys=True,
                        ),
                    ),
                    max_tokens=3072,
                )
            except Exception as exc:
                first_pass = _default_first_pass("")
                _write_checkpoint(
                    checkpoint_path,
                    _checkpoint_state(
                        video_path,
                        metadata,
                        first_pass,
                        detail_passes,
                        [],
                        seen_windows,
                        len(coarse_frames),
                        budget,
                        {
                            **sampling_strategy,
                            "coarse_sampled_timestamps_sec": [
                                frame["timestamp_sec"] for frame in coarse_frames
                            ],
                        },
                        "failed",
                        error=_error_message("coarse_pass", exc),
                    ),
                )
                return _build_result(
                    video_path=video_path,
                    metadata=metadata,
                    first_pass=first_pass,
                    detail_passes=detail_passes,
                    final={},
                    final_raw="",
                    budget=budget,
                    sampled_frames=len(coarse_frames),
                    sampling_strategy={
                        **sampling_strategy,
                        "coarse_sampled_timestamps_sec": [
                            frame["timestamp_sec"] for frame in coarse_frames
                        ],
                    },
                    unfilled_focus_requests=[],
                    analysis_status="failed",
                    error=_error_message("coarse_pass", exc),
                    checkpoint_path=checkpoint_path,
                )
            first_pass = _parse_json_response(
                first_raw,
                default=_default_first_pass(first_raw),
            )
            sampled_frames = len(coarse_frames)
            focus_queue = _normalize_focus_requests(
                first_pass.get("focus_requests", []),
                duration_sec=metadata.get("duration_sec", 0),
            )
            focus_queue = _merge_focus_queue(
                focus_queue,
                seen_windows=seen_windows,
                duration_sec=metadata.get("duration_sec", 0),
            )
            sampling_strategy["coarse_sampled_timestamps_sec"] = [
                frame["timestamp_sec"] for frame in coarse_frames
            ]
            _write_checkpoint(
                checkpoint_path,
                _checkpoint_state(
                    video_path,
                    metadata,
                    first_pass,
                    detail_passes,
                    focus_queue,
                    seen_windows,
                    sampled_frames,
                    budget,
                    sampling_strategy,
                    "coarse_complete",
                ),
            )

        while (
            focus_queue
            and len(detail_passes) < max_focus_windows
            and budget.remaining > 1
            and sampled_frames < max_sampled_frames
        ):
            window = focus_queue.pop(0)
            key = (round(window["start_sec"], 1), round(window["end_sec"], 1))
            if key in seen_windows or _overlaps_seen_windows(window, seen_windows):
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
            try:
                detail_raw = self.vlm.analyze_frames(
                    frames,
                    DETAIL_PASS_PROMPT.replace(
                        "{window_json}", json.dumps(window, indent=2, sort_keys=True)
                    ).replace(
                        "{budget_json}",
                        json.dumps(
                            _budget_context(
                                budget=budget,
                                sampled_frames=sampled_frames,
                                max_sampled_frames=max_sampled_frames,
                                detail_passes=detail_passes,
                                max_focus_windows=max_focus_windows,
                                frames_per_focus=frames_per_focus,
                                reviewed_windows=[
                                    item["requested_window"] for item in detail_passes
                                ],
                            ),
                            indent=2,
                            sort_keys=True,
                        ),
                    ),
                    max_tokens=3072,
                )
            except Exception as exc:
                unfilled = [window, *focus_queue]
                _write_checkpoint(
                    checkpoint_path,
                    _checkpoint_state(
                        video_path,
                        metadata,
                        first_pass,
                        detail_passes,
                        unfilled,
                        seen_windows,
                        sampled_frames,
                        budget,
                        sampling_strategy,
                        "failed",
                        error=_error_message("detail_pass", exc),
                    ),
                )
                return _build_result(
                    video_path=video_path,
                    metadata=metadata,
                    first_pass=first_pass,
                    detail_passes=detail_passes,
                    final={},
                    final_raw="",
                    budget=budget,
                    sampled_frames=sampled_frames,
                    sampling_strategy=sampling_strategy,
                    unfilled_focus_requests=unfilled,
                    analysis_status="failed",
                    error=_error_message("detail_pass", exc),
                    checkpoint_path=checkpoint_path,
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
                focus_queue = _merge_focus_queue(
                    [*focus_queue, *followups],
                    seen_windows=seen_windows,
                    duration_sec=metadata.get("duration_sec", 0),
                )
            _write_checkpoint(
                checkpoint_path,
                _checkpoint_state(
                    video_path,
                    metadata,
                    first_pass,
                    detail_passes,
                    focus_queue,
                    seen_windows,
                    sampled_frames,
                    budget,
                    sampling_strategy,
                    "detail_complete",
                ),
            )

        synthesis_input = {
            "video_metadata": metadata,
            "first_pass": first_pass,
            "detail_passes": [
                {
                    "requested_window": item["requested_window"],
                    "sampled_timestamps_sec": item["sampled_timestamps_sec"],
                        "analysis": _compact_detail_analysis(item["analysis"]),
                }
                for item in detail_passes
            ],
        }

        final_raw = ""
        if budget.remaining > 0:
            budget.spend()
            try:
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
            except Exception as exc:
                _write_checkpoint(
                    checkpoint_path,
                    _checkpoint_state(
                        video_path,
                        metadata,
                        first_pass,
                        detail_passes,
                        focus_queue,
                        seen_windows,
                        sampled_frames,
                        budget,
                        sampling_strategy,
                        "failed",
                        error=_error_message("final_synthesis", exc),
                    ),
                )
                return _build_result(
                    video_path=video_path,
                    metadata=metadata,
                    first_pass=first_pass,
                    detail_passes=detail_passes,
                    final={},
                    final_raw="",
                    budget=budget,
                    sampled_frames=sampled_frames,
                    sampling_strategy=sampling_strategy,
                    unfilled_focus_requests=focus_queue,
                    analysis_status="partial",
                    error=_error_message("final_synthesis", exc),
                    checkpoint_path=checkpoint_path,
                )
            final = _parse_json_response(final_raw, default={})
        else:
            final = {}

        final_truncated = bool(final_raw) and not _looks_like_complete_json_response(final_raw)
        final = _final_with_fallbacks(final, first_pass, detail_passes)
        analysis_status = _analysis_status(
            final_truncated=final_truncated,
            focus_queue=focus_queue,
            budget=budget,
        )
        error = "final_synthesis: response was truncated or incomplete" if final_truncated else None

        result = _build_result(
            video_path=video_path,
            metadata=metadata,
            first_pass=first_pass,
            detail_passes=detail_passes,
            final=final,
            final_raw=final_raw,
            budget=budget,
            sampled_frames=sampled_frames,
            sampling_strategy=sampling_strategy,
            unfilled_focus_requests=focus_queue,
            analysis_status=analysis_status,
            error=error,
            checkpoint_path=checkpoint_path,
        )
        _write_checkpoint(
            checkpoint_path,
            {
                **_checkpoint_state(
                    video_path,
                    metadata,
                    first_pass,
                    detail_passes,
                    focus_queue,
                    seen_windows,
                    sampled_frames,
                    budget,
                    sampling_strategy,
                    "complete",
                ),
                "final": final,
            },
        )
        return result


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


def _load_checkpoint(path: str | None) -> dict:
    if not path:
        return {}
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {}
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_checkpoint(path: str | None, state: dict) -> None:
    if not path:
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _checkpoint_state(
    video_path: str,
    metadata: dict,
    first_pass: dict,
    detail_passes: list[dict],
    focus_queue: list[dict],
    seen_windows: set[tuple[float, float]],
    sampled_frames: int,
    budget: RequestBudget,
    sampling_strategy: dict,
    status: str,
    error: str | None = None,
) -> dict:
    return {
        "type": "lab_review_checkpoint",
        "status": status,
        "video_path": video_path,
        "metadata": metadata,
        "first_pass": first_pass,
        "detail_passes": detail_passes,
        "focus_queue": focus_queue,
        "seen_windows": [list(item) for item in sorted(seen_windows)],
        "sampled_frames": sampled_frames,
        "request_budget": budget.as_dict(),
        "sampling_strategy": sampling_strategy,
        "error": error,
    }


def _error_message(stage: str, exc: Exception) -> str:
    return f"{stage}: {type(exc).__name__}: {exc}"


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


def _merge_focus_queue(
    items: list[dict],
    seen_windows: set[tuple[float, float]],
    duration_sec: float,
    overlap_threshold: float = 0.6,
) -> list[dict]:
    normalized = _normalize_focus_requests(items, duration_sec)
    merged: list[dict] = []
    for item in normalized:
        if _overlaps_seen_windows(item, seen_windows, overlap_threshold):
            continue
        if any(_window_overlap_ratio(item, existing) >= overlap_threshold for existing in merged):
            existing = next(
                existing
                for existing in merged
                if _window_overlap_ratio(item, existing) >= overlap_threshold
            )
            existing["start_sec"] = round(
                min(existing["start_sec"], item["start_sec"]),
                2,
            )
            existing["end_sec"] = round(max(existing["end_sec"], item["end_sec"]), 2)
            existing["reason"] = f"{existing['reason']} / {item['reason']}"
            existing["priority"] = _higher_priority(existing["priority"], item["priority"])
            continue
        merged.append(item)
    return _normalize_focus_requests(merged, duration_sec)


def _overlaps_seen_windows(
    window: dict,
    seen_windows: set[tuple[float, float]],
    overlap_threshold: float = 0.6,
) -> bool:
    return any(
        _window_overlap_ratio(
            window,
            {"start_sec": start, "end_sec": end},
        )
        >= overlap_threshold
        for start, end in seen_windows
    )


def _window_overlap_ratio(left: dict, right: dict) -> float:
    left_start = float(left.get("start_sec") or 0)
    left_end = float(left.get("end_sec") or left_start)
    right_start = float(right.get("start_sec") or 0)
    right_end = float(right.get("end_sec") or right_start)
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shorter = max(0.01, min(left_end - left_start, right_end - right_start))
    return overlap / shorter


def _higher_priority(left: str, right: str) -> str:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return min(
        [left, right],
        key=lambda value: priority_order.get(str(value).lower(), 1),
    )


def _budget_context(
    budget: RequestBudget,
    sampled_frames: int,
    max_sampled_frames: int,
    detail_passes: list[dict],
    max_focus_windows: int,
    frames_per_focus: int,
    reviewed_windows: list[dict],
) -> dict:
    remaining_frame_budget = max(0, max_sampled_frames - sampled_frames)
    remaining_detail_windows = max(0, max_focus_windows - len(detail_passes))
    remaining_detail_requests = max(0, budget.remaining - 1)
    possible_detail_windows = min(remaining_detail_windows, remaining_detail_requests)
    return {
        "remaining_claude_requests_after_this_request": max(0, budget.remaining - 1),
        "reserve_one_request_for_final_synthesis": True,
        "remaining_detail_windows_after_this_request": possible_detail_windows,
        "remaining_frame_budget_after_this_request": remaining_frame_budget,
        "frames_per_focus_window": frames_per_focus,
        "already_reviewed_windows": reviewed_windows,
        "instruction": (
            "Request only non-overlapping high-value windows. Most lab footage "
            "is low information; spend follow-up windows on visible actions, "
            "labels, measurements, transfers, instrument displays, and moments "
            "where reproducibility depends on tacit details."
        ),
    }


def _compact_detail_analysis(analysis: dict) -> dict:
    return {
        "observed_actions": _top_items(analysis.get("observed_actions") or [], 4),
        "reproducibility_risks": _top_items(
            analysis.get("reproducibility_risks") or [],
            4,
            severity_sort=True,
        ),
        "thumbs_up": _top_items(analysis.get("thumbs_up") or [], 3),
        "notes": analysis.get("notes", ""),
    }


def _top_items(items: list[dict], limit: int, severity_sort: bool = False) -> list[dict]:
    if not severity_sort:
        return items[:limit]
    severity_order = {"Very High": 0, "High": 1, "Medium": 2, "Low": 3}
    return sorted(
        items,
        key=lambda item: (
            severity_order.get(item.get("severity"), 4),
            _to_float(item.get("timestamp_sec")) or 0,
        ),
    )[:limit]


def _looks_like_complete_json_response(text: str) -> bool:
    parsed = _parse_json_response(text, default={})
    if not parsed:
        return False
    required = {"review_summary", "event_timeline", "reproducibility_metrics", "meta_advice"}
    return required.issubset(parsed.keys())


def _final_with_fallbacks(
    final: dict,
    first_pass: dict,
    detail_passes: list[dict],
) -> dict:
    final = dict(final or {})
    risks = _merge_detail_items(detail_passes, "reproducibility_risks")
    actions = _merge_detail_items(detail_passes, "observed_actions")
    good = _merge_detail_items(detail_passes, "thumbs_up")

    final.setdefault("review_summary", first_pass.get("video_summary", ""))
    final.setdefault("event_timeline", first_pass.get("event_timeline", []))
    final.setdefault("observed_actions", actions)
    final.setdefault("reproducibility_risks", risks)
    final.setdefault("thumbs_up", good)
    if not final.get("reproducibility_metrics"):
        final["reproducibility_metrics"] = _fallback_metrics(risks, actions)
    if not final.get("meta_advice"):
        final["meta_advice"] = _fallback_meta_advice(risks, first_pass)
    return final


def _fallback_metrics(risks: list[dict], actions: list[dict]) -> list[dict]:
    risk_text = " ".join(
        f"{risk.get('action', '')} {risk.get('issue', '')}" for risk in risks
    ).lower()
    action_text = " ".join(action.get("action", "") for action in actions).lower()
    return [
        _metric(
            "critical-parameters",
            1 if any(term in risk_text for term in ["volume", "cell line", "passage", "label"]) else 2,
            "Critical identities, labels, volumes, or passage details were frequently not visible.",
            "Show labels and pipette settings; verbally state cell line, passage, reagent, and volume before each critical action.",
        ),
        _metric(
            "materials-identification",
            2 if risks else 3,
            "Equipment was identifiable, but specific reagents/samples were often unclear.",
            "Stage materials with labels facing the camera and capture a short pre-run inventory.",
        ),
        _metric(
            "action-order",
            3 if actions else 1,
            "The broad order of movement through the lab is visible, but exact manipulations are intermittent.",
            "Keep the camera fixed on the active work area during transitions and state each step before performing it.",
        ),
        _metric(
            "measurement-capture",
            1 if "display" in risk_text or "setting" in risk_text else 2,
            "Quantitative settings/readouts were not consistently legible.",
            "Pause on instrument displays, incubator settings, timers, and pipette dials long enough for capture.",
        ),
        _metric(
            "timing-capture",
            2,
            "Video timestamps exist, but procedural timing and incubation durations were not explicitly documented.",
            "Announce start/stop times for incubations, BSC work, and reagent exposure windows.",
        ),
    ]


def _metric(metric: str, score: int, evidence: str, recommendation: str) -> dict:
    return {
        "metric": metric,
        "score": score,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _fallback_meta_advice(risks: list[dict], first_pass: dict) -> dict:
    top_fixes = []
    for risk in risks:
        fix = risk.get("suggested_fix")
        if fix and fix not in top_fixes:
            top_fixes.append(fix)
        if len(top_fixes) >= 5:
            break
    if not top_fixes:
        top_fixes = [
            "Use a stable camera angle aimed at the active work area.",
            "Verbally state sample IDs, reagent names, volumes, and timing.",
            "Capture labels and instrument displays close enough to read.",
        ]
    return {
        "overall_review": first_pass.get("video_summary", "")
        or "The video captured broad workflow context, but key reproducibility details require clearer capture.",
        "next_time": top_fixes,
    }


def _analysis_status(
    final_truncated: bool,
    focus_queue: list[dict],
    budget: RequestBudget,
) -> str:
    if final_truncated:
        return "partial"
    high_priority_remaining = any(
        str(item.get("priority", "")).lower() == "high" for item in focus_queue
    )
    if high_priority_remaining or (focus_queue and budget.remaining == 0):
        return "partial"
    return "complete"


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
    analysis_status: str = "complete",
    error: str | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    merged_actions = _merge_detail_items(detail_passes, "observed_actions")
    merged_risks = _merge_detail_items(detail_passes, "reproducibility_risks")
    merged_good = _merge_detail_items(detail_passes, "thumbs_up")

    protocol = final.get("protocol") or {}
    return {
        "task": "lab_review",
        "analysis_status": analysis_status,
        "error": error,
        "checkpoint_path": checkpoint_path,
        "video_path": video_path,
        "video_metadata": metadata,
        "review_summary": final.get("review_summary")
        or first_pass.get("video_summary", ""),
        "event_timeline": final.get("event_timeline")
        or first_pass.get("event_timeline", []),
        "observed_actions": merged_actions or final.get("observed_actions", []),
        "reproducibility_risks": merged_risks
        or final.get("reproducibility_risks", []),
        "thumbs_up": merged_good or final.get("thumbs_up", []),
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
