"""Video frame extraction utilities for lab video analysis."""

import base64
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


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


def extract_frames_from_video_set(
    video_source: str,
    interval_seconds: float = 5.0,
    max_frames: int | None = None,
    resize_width: int | None = 1024,
) -> list[dict]:
    """Extract sparse frames from a file or folder as one virtual timeline."""
    video_set = get_video_set_metadata(video_source)
    timestamps = _uniform_timestamps(
        duration_sec=video_set["duration_sec"],
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )
    return extract_frames_at_global_timestamps(
        video_source,
        timestamps,
        resize_width=resize_width,
        video_set=video_set,
    )


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


def extract_frames_at_global_timestamps(
    video_source: str,
    timestamps_sec: list[float],
    resize_width: int | None = 1024,
    video_set: dict | None = None,
) -> list[dict]:
    """Extract frames from file/folder using global timestamps."""
    video_set = video_set or get_video_set_metadata(video_source)
    chunks = video_set["chunks"]
    frames = []

    for timestamp in sorted(timestamps_sec):
        chunk = _chunk_for_global_timestamp(chunks, timestamp)
        if chunk is None:
            continue
        local_ts = max(0.0, timestamp - chunk["start_sec"])
        if local_ts >= chunk["duration_sec"]:
            local_ts = max(0.0, chunk["duration_sec"] - 0.01)
        local_frames = extract_frames_at_timestamps(
            chunk["path"],
            [local_ts],
            resize_width=resize_width,
        )
        if not local_frames:
            continue
        frame = local_frames[0]
        extracted_local_ts = frame.get("timestamp_sec", local_ts)
        frame["timestamp_sec"] = round(timestamp, 2)
        frame["local_timestamp_sec"] = round(extracted_local_ts, 2)
        frame["source_video_path"] = chunk["path"]
        frame["source_video_index"] = chunk["index"]
        frame["source_video_start_sec"] = chunk["start_sec"]
        frames.append(frame)

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


def get_video_set_metadata(video_source: str) -> dict:
    """Return metadata for a single video file or sorted folder of video chunks."""
    paths = list_video_files(video_source)
    chunks = []
    timeline_start = 0.0
    total_frames = 0
    width = None
    height = None
    fps = None

    for index, path in enumerate(paths):
        metadata = get_video_metadata(str(path))
        duration = float(metadata.get("duration_sec") or 0)
        chunks.append(
            {
                "index": index,
                "path": str(path),
                "filename": path.name,
                "start_sec": round(timeline_start, 2),
                "end_sec": round(timeline_start + duration, 2),
                "duration_sec": duration,
                "fps": metadata.get("fps"),
                "total_frames": metadata.get("total_frames"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
            }
        )
        timeline_start += duration
        total_frames += int(metadata.get("total_frames") or 0)
        width = max(width or 0, int(metadata.get("width") or 0))
        height = max(height or 0, int(metadata.get("height") or 0))
        fps = fps or metadata.get("fps")

    return {
        "source": str(Path(video_source)),
        "is_video_set": len(paths) > 1 or Path(video_source).is_dir(),
        "chunk_count": len(paths),
        "chunks": chunks,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(timeline_start, 2),
        "width": width or 0,
        "height": height or 0,
    }


def list_video_files(video_source: str) -> list[Path]:
    """Return a sorted list of video files from a file or directory source."""
    source = Path(video_source)
    if not source.exists():
        raise FileNotFoundError(f"Video source not found: {video_source}")
    if source.is_file():
        return [source]
    paths = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No video files found in folder: {video_source}")
    return paths


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


def _uniform_timestamps(
    duration_sec: float,
    interval_seconds: float,
    max_frames: int | None,
) -> list[float]:
    interval = max(0.25, float(interval_seconds or 1.0))
    timestamps = []
    current = 0.0
    while current <= duration_sec:
        timestamps.append(round(current, 2))
        if max_frames is not None and len(timestamps) >= max_frames:
            break
        current += interval
    return timestamps


def _chunk_for_global_timestamp(chunks: list[dict], timestamp_sec: float) -> dict | None:
    for chunk in chunks:
        if chunk["start_sec"] <= timestamp_sec < chunk["end_sec"]:
            return chunk
    if chunks and timestamp_sec >= chunks[-1]["end_sec"]:
        return chunks[-1]
    return None
