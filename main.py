"""Lab Video Review CLI entry point.

Usage examples
--------------
# Budgeted Claude review with sparse first pass and requested detail windows:
python main.py --video path/to/video.mp4 --task lab_review --max-claude-requests 6

# Write viewer-compatible annotations:
python main.py --video path/to/video.mp4 --output review.annotations.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

# Load .env file if present (for API keys)
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Budgeted lab video review using Claude-only sparse frame sampling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video",
        required=True,
        metavar="PATH",
        help="Path to the lab video file.",
    )
    parser.add_argument(
        "--task",
        default="lab_review",
        choices=[
            "lab_review",
            "well_plate",
            "color_change",
            "od_values",
            "yeast_protocol",
            "volume",
            "protocol",
            "all",
        ],
        help=(
            "Analysis task to run:\n"
            "  lab_review   – budgeted sparse Claude review for lab videos\n"
            "  well_plate   – Video A: count wells pipetted in 96-well plate\n"
            "  color_change – Video B: detect color changes over time\n"
            "  od_values    – Video C: read OD values from display\n"
            "  yeast_protocol – Video C: draft yeast transformation protocol\n"
            "  volume       – Video D: read liquid volume from glassware\n"
            "  protocol     – Stretch goal: write full lab protocol\n"
            "  all          – Run all tasks then write a protocol"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["claude"],
        default="claude",
        help="VLM provider to use. This migrated workflow is Claude-only.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model name to use (overrides provider default).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON results to this file (default: print to stdout).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override legacy task interval or lab_review coarse interval.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Override legacy max frames or lab_review coarse frame cap.",
    )
    parser.add_argument(
        "--coarse-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="lab_review first-pass sampling interval (default: 30).",
    )
    parser.add_argument(
        "--coarse-max-frames",
        type=int,
        default=None,
        metavar="N",
        help="lab_review first-pass frame cap (default: 18).",
    )
    parser.add_argument(
        "--detail-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="lab_review frame interval inside Claude-requested windows.",
    )
    parser.add_argument(
        "--frames-per-focus",
        type=int,
        default=None,
        metavar="N",
        help="lab_review frame cap per requested detail window.",
    )
    parser.add_argument(
        "--max-focus-windows",
        type=int,
        default=None,
        metavar="N",
        help="lab_review maximum number of detail windows to inspect.",
    )
    parser.add_argument(
        "--max-claude-requests",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Hard cap on Claude requests for lab_review, including first pass "
            "and final synthesis (default: 6)."
        ),
    )
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=None,
        metavar="N",
        help="Hard cap on total video frames uploaded to Claude for lab_review.",
    )
    return parser


def run_task(task: str, vlm_client, video_path: str, extra_kwargs: dict) -> dict:
    """Dispatch to the appropriate analyzer and return results."""
    from src.analyzers.well_plate_analyzer import WellPlateAnalyzer
    from src.analyzers.color_change_analyzer import ColorChangeAnalyzer
    from src.analyzers.od_value_analyzer import ODValueAnalyzer
    from src.analyzers.yeast_transformation_analyzer import YeastTransformationAnalyzer
    from src.analyzers.volume_analyzer import VolumeAnalyzer
    from src.analyzers.protocol_writer import ProtocolWriter
    from src.analyzers.lab_review_analyzer import LabReviewAnalyzer

    analyzers = {
        "lab_review": LabReviewAnalyzer,
        "well_plate": WellPlateAnalyzer,
        "color_change": ColorChangeAnalyzer,
        "od_values": ODValueAnalyzer,
        "yeast_protocol": YeastTransformationAnalyzer,
        "volume": VolumeAnalyzer,
    }
    challenge_analyzers = {
        name: analyzers[name]
        for name in ("well_plate", "color_change", "od_values", "volume")
    }

    if task == "protocol":
        return ProtocolWriter(vlm_client).analyze(video_path, **extra_kwargs)

    if task == "all":
        all_results = {}
        for name, cls in challenge_analyzers.items():
            print(f"  Running {name} ...", file=sys.stderr)
            all_results[name] = cls(vlm_client).analyze(video_path, **extra_kwargs)
        print("  Writing protocol ...", file=sys.stderr)
        protocol = ProtocolWriter(vlm_client).analyze(
            video_path, extracted_results=all_results, **extra_kwargs
        )
        return {"individual_analyses": all_results, "protocol": protocol}

    return analyzers[task](vlm_client).analyze(video_path, **extra_kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    video_path = args.video
    if not Path(video_path).exists():
        print(f"ERROR: video file not found: {video_path}", file=sys.stderr)
        return 1

    # Build VLM client
    from src.vlm_client import VLMClient, VLMProvider

    try:
        vlm = VLMClient(provider=args.provider, model=args.model)
    except EnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Build optional kwargs
    extra: dict = {}
    if args.task == "lab_review":
        if args.interval is not None:
            extra["coarse_interval_seconds"] = args.interval
        if args.max_frames is not None:
            extra["coarse_max_frames"] = args.max_frames
        if args.coarse_interval is not None:
            extra["coarse_interval_seconds"] = args.coarse_interval
        if args.coarse_max_frames is not None:
            extra["coarse_max_frames"] = args.coarse_max_frames
        if args.detail_interval is not None:
            extra["detail_interval_seconds"] = args.detail_interval
        if args.frames_per_focus is not None:
            extra["frames_per_focus"] = args.frames_per_focus
        if args.max_focus_windows is not None:
            extra["max_focus_windows"] = args.max_focus_windows
        if args.max_claude_requests is not None:
            extra["max_claude_requests"] = args.max_claude_requests
        if args.max_sampled_frames is not None:
            extra["max_sampled_frames"] = args.max_sampled_frames
    elif args.interval is not None:
        extra["interval_seconds"] = args.interval
    if args.task != "lab_review" and args.max_frames is not None:
        extra["max_frames"] = args.max_frames

    print(
        f"Running task '{args.task}' on '{video_path}' with {args.provider} ...",
        file=sys.stderr,
    )

    try:
        results = run_task(args.task, vlm, video_path, extra)
    except Exception as exc:
        print(f"ERROR during analysis: {exc}", file=sys.stderr)
        raise

    output_str = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
