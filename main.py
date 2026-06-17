"""PRISM Lab Video Analyzer – CLI entry point.

Usage examples
--------------
# Analyze Video A (well plate counting) with Claude:
python main.py --video path/to/video_a.mp4 --task well_plate

# Analyze Video B (color changes) with Gemini:
python main.py --video path/to/video_b.mp4 --task color_change --provider gemini

# Analyze Video C (OD values):
python main.py --video path/to/video_c.mp4 --task od_values

# Analyze Video D (liquid volume):
python main.py --video path/to/video_d.mp4 --task volume

# Stretch goal – write a full protocol:
python main.py --video path/to/video.mp4 --task protocol

# Run all tasks and write a protocol:
python main.py --video path/to/video.mp4 --task all
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (for API keys)
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PRISM Lab Video Analyzer – extract protocol information "
                    "from lab videos using VLM (Claude / Gemini).",
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
        required=True,
        choices=["well_plate", "color_change", "od_values", "volume", "protocol", "all"],
        help=(
            "Analysis task to run:\n"
            "  well_plate   – Video A: count wells pipetted in 96-well plate\n"
            "  color_change – Video B: detect color changes over time\n"
            "  od_values    – Video C: read OD values from display\n"
            "  volume       – Video D: read liquid volume from glassware\n"
            "  protocol     – Stretch goal: write full lab protocol\n"
            "  all          – Run all tasks then write a protocol"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "gemini"],
        default="claude",
        help="VLM provider to use (default: claude).",
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
        help="Override the frame sampling interval (seconds).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Override the maximum number of frames sent to the VLM.",
    )
    return parser


def run_task(task: str, vlm_client, video_path: str, extra_kwargs: dict) -> dict:
    """Dispatch to the appropriate analyzer and return results."""
    from src.analyzers.well_plate_analyzer import WellPlateAnalyzer
    from src.analyzers.color_change_analyzer import ColorChangeAnalyzer
    from src.analyzers.od_value_analyzer import ODValueAnalyzer
    from src.analyzers.volume_analyzer import VolumeAnalyzer
    from src.analyzers.protocol_writer import ProtocolWriter

    analyzers = {
        "well_plate": WellPlateAnalyzer,
        "color_change": ColorChangeAnalyzer,
        "od_values": ODValueAnalyzer,
        "volume": VolumeAnalyzer,
    }

    if task == "protocol":
        return ProtocolWriter(vlm_client).analyze(video_path, **extra_kwargs)

    if task == "all":
        all_results = {}
        for name, cls in analyzers.items():
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
    if args.interval is not None:
        extra["interval_seconds"] = args.interval
    if args.max_frames is not None:
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
