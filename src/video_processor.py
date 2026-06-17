"""Video frame extraction utilities for lab video analysis."""

import base64
from pathlib import Path


def extract_frames(
    video_path: str,
    interval_seconds: float = 5.0,
    max_frames: int | None = None,
    resize_width: int | None = 1024,
) -> list[dict]:
    """Extract frames from a video at a specified time interval.

    Args:
        video_path: Path to the video file.
        interval_seconds: Time between extracted frames in seconds.
        max_frames: Maximum number of frames to extract (None = all).
        resize_width: Resize frames to this width while preserving aspect ratio
                      (None = no resize).

    Returns:
        List of dicts with keys: 'timestamp_sec', 'frame_index', 'image_b64'.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    frame_step = max(1, int(fps * interval_seconds))
    frames = []
    frame_index = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_index / fps if fps > 0 else 0
        image_b64 = _frame_to_base64(frame, resize_width)

        frames.append(
            {
                "timestamp_sec": round(timestamp_sec, 2),
                "frame_index": frame_index,
                "image_b64": image_b64,
            }
        )

        if max_frames is not None and len(frames) >= max_frames:
            break

        frame_index += frame_step

    cap.release()
    return frames


def extract_frames_at_timestamps(
    video_path: str,
    timestamps_sec: list[float],
    resize_width: int | None = 1024,
) -> list[dict]:
    """Extract frames at specific timestamps.

    Args:
        video_path: Path to the video file.
        timestamps_sec: List of timestamps (seconds) to extract.
        resize_width: Resize frames to this width while preserving aspect ratio.

    Returns:
        List of dicts with keys: 'timestamp_sec', 'frame_index', 'image_b64'.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    for ts in sorted(timestamps_sec):
        frame_index = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            continue

        image_b64 = _frame_to_base64(frame, resize_width)
        frames.append(
            {
                "timestamp_sec": round(ts, 2),
                "frame_index": frame_index,
                "image_b64": image_b64,
            }
        )

    cap.release()
    return frames


def get_video_metadata(video_path: str) -> dict:
    """Return basic metadata about a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with 'fps', 'total_frames', 'duration_sec', 'width', 'height'.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
        "width": width,
        "height": height,
    }


def _frame_to_base64(frame: object, resize_width: int | None) -> str:
    """Convert an OpenCV frame (BGR) to a base64-encoded JPEG string."""
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if resize_width is not None:
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
