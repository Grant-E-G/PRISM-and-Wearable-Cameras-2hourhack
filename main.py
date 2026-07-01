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
import math
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

# Load .env file if present (for API keys)
load_dotenv()


MODEL_PRICING_USD_PER_MTOK = {
    # Prices are API list prices checked against Anthropic pricing on 2026-07-01.
    "haiku": {"input": 1.00, "output": 5.00},
    "sonnet-5": {"input": 2.00, "output": 10.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus-4-1": {"input": 15.00, "output": 75.00},
    "opus": {"input": 5.00, "output": 25.00},
    "fable": {"input": 10.00, "output": 50.00},
    "mythos": {"input": 10.00, "output": 50.00},
}
HIGH_RES_MODEL_MARKERS = (
    "fable-5",
    "mythos-5",
    "opus-4-7",
    "opus-4-8",
    "sonnet-5",
)
DEFAULT_CLAUDE_MODEL = "claude-opus-4-5"
DEFAULT_LAB_REVIEW_OPTIONS = {
    "coarse_max_frames": 18,
    "frames_per_focus": 8,
    "max_focus_windows": 6,
    "max_claude_requests": 6,
    "max_sampled_frames": 70,
}
FIRST_PASS_OUTPUT_TOKENS = 3072
DETAIL_PASS_OUTPUT_TOKENS = 3072
FINAL_SYNTHESIS_OUTPUT_TOKENS = 4096
DEFAULT_RESIZE_WIDTH = 1024


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


def estimate_lab_review_cost(video_path: str, model: str | None, options: dict) -> dict:
    """Estimate the configured worst-case Claude cost before uploading frames."""
    from src.video_processor import get_video_metadata

    metadata = get_video_metadata(video_path)
    effective_options = {**DEFAULT_LAB_REVIEW_OPTIONS, **options}
    model_name = model or DEFAULT_CLAUDE_MODEL
    pricing = _pricing_for_model(model_name)

    resized_width, resized_height = _resized_dimensions(
        width=int(metadata.get("width") or 0),
        height=int(metadata.get("height") or 0),
        resize_width=DEFAULT_RESIZE_WIDTH,
    )
    visual_tokens_per_frame = _visual_tokens_for_image(
        resized_width,
        resized_height,
        model_name,
    )
    max_frames = _max_uploaded_frames(effective_options)
    max_requests = int(effective_options["max_claude_requests"])
    output_token_ceiling = _max_output_tokens(effective_options)

    image_input_tokens = visual_tokens_per_frame * max_frames
    image_input_cost = image_input_tokens / 1_000_000 * pricing["input"]
    output_cost_ceiling = output_token_ceiling / 1_000_000 * pricing["output"]

    return {
        "model": model_name,
        "pricing": pricing,
        "video": {
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "duration_sec": metadata.get("duration_sec"),
        },
        "resized_frame": {
            "width": resized_width,
            "height": resized_height,
            "visual_tokens": visual_tokens_per_frame,
        },
        "max_uploaded_frames": max_frames,
        "max_claude_requests": max_requests,
        "image_input_tokens": image_input_tokens,
        "output_token_ceiling": output_token_ceiling,
        "image_input_cost_usd": image_input_cost,
        "output_cost_ceiling_usd": output_cost_ceiling,
        "estimated_ceiling_usd": image_input_cost + output_cost_ceiling,
        "notes": (
            "Estimate includes image input tokens and configured maximum output "
            "tokens. Text prompt input, final synthesis text input, taxes, and "
            "provider-side pricing changes are not included."
        ),
    }


def confirm_lab_review_cost(cost: dict) -> bool:
    """Print the cost estimate and require y/yes before continuing."""
    video = cost["video"]
    frame = cost["resized_frame"]
    pricing = cost["pricing"]
    print("\nClaude cost estimate for this lab_review run:", file=sys.stderr)
    print(f"  Model: {cost['model']}", file=sys.stderr)
    print(
        "  Pricing used: "
        f"${pricing['input']:.2f}/M input tokens, "
        f"${pricing['output']:.2f}/M output tokens",
        file=sys.stderr,
    )
    print(
        "  Video: "
        f"{video['width']}x{video['height']}, "
        f"{video['duration_sec']}s",
        file=sys.stderr,
    )
    print(
        "  Uploaded frame size estimate: "
        f"{frame['width']}x{frame['height']} "
        f"({frame['visual_tokens']} visual tokens/frame)",
        file=sys.stderr,
    )
    print(
        "  Configured caps: "
        f"{cost['max_claude_requests']} Claude requests, "
        f"{cost['max_uploaded_frames']} uploaded frames",
        file=sys.stderr,
    )
    print(
        "  Image input ceiling: "
        f"{cost['image_input_tokens']:,} tokens "
        f"= ${cost['image_input_cost_usd']:.4f}",
        file=sys.stderr,
    )
    print(
        "  Output token ceiling: "
        f"{cost['output_token_ceiling']:,} tokens "
        f"= ${cost['output_cost_ceiling_usd']:.4f}",
        file=sys.stderr,
    )
    print(
        f"  Estimated ceiling before text input: "
        f"${cost['estimated_ceiling_usd']:.4f}",
        file=sys.stderr,
    )
    print(f"  Note: {cost['notes']}", file=sys.stderr)

    try:
        answer = input("Continue and run Claude analysis? Type y to continue: ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


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

    if args.task == "lab_review":
        cost = estimate_lab_review_cost(video_path, args.model, extra)
        if not confirm_lab_review_cost(cost):
            print("Aborted before Claude analysis.", file=sys.stderr)
            return 1

    # Build VLM client only after the cost gate has been accepted.
    from src.vlm_client import VLMClient

    try:
        vlm = VLMClient(provider=args.provider, model=args.model)
    except EnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

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


def _pricing_for_model(model: str) -> dict:
    normalized = model.lower().replace("_", "-")
    if "opus-4-1" in normalized:
        return MODEL_PRICING_USD_PER_MTOK["opus-4-1"]
    for marker in ("sonnet-5", "haiku", "sonnet", "opus", "fable", "mythos"):
        if marker in normalized:
            return MODEL_PRICING_USD_PER_MTOK[marker]
    # Unknown Claude models should fail conservative instead of underestimating.
    return MODEL_PRICING_USD_PER_MTOK["opus"]


def _resized_dimensions(width: int, height: int, resize_width: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return resize_width, resize_width
    if width <= resize_width:
        return width, height
    scale = resize_width / width
    return resize_width, max(1, int(height * scale))


def _visual_tokens_for_image(width: int, height: int, model: str) -> int:
    raw_tokens = math.ceil(width / 28) * math.ceil(height / 28)
    max_tokens = 4784 if _uses_high_resolution_tier(model) else 1568
    return min(raw_tokens, max_tokens)


def _uses_high_resolution_tier(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return any(marker in normalized for marker in HIGH_RES_MODEL_MARKERS)


def _max_uploaded_frames(options: dict) -> int:
    max_sampled_frames = int(options["max_sampled_frames"])
    coarse_frames = min(int(options["coarse_max_frames"]), max_sampled_frames)
    max_requests = int(options["max_claude_requests"])
    if max_requests <= 2:
        return coarse_frames
    detail_requests = min(
        int(options["max_focus_windows"]),
        max(0, max_requests - 2),
    )
    detail_frames = detail_requests * int(options["frames_per_focus"])
    return min(max_sampled_frames, coarse_frames + detail_frames)


def _max_output_tokens(options: dict) -> int:
    max_requests = int(options["max_claude_requests"])
    total = FIRST_PASS_OUTPUT_TOKENS
    if max_requests >= 2:
        total += FINAL_SYNTHESIS_OUTPUT_TOKENS
    detail_requests = min(
        int(options["max_focus_windows"]),
        max(0, max_requests - 2),
    )
    return total + detail_requests * DETAIL_PASS_OUTPUT_TOKENS


if __name__ == "__main__":
    sys.exit(main())
