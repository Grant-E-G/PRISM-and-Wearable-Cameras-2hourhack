"""Static file server with byte-range support for browser video seeking."""

from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class _RangeFile:
    """File wrapper that stops reads after the requested byte count."""

    def __init__(self, file_obj, remaining: int) -> None:
        self.file_obj = file_obj
        self.remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        chunk = self.file_obj.read(size)
        self.remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self.file_obj.close()


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """Serve normal static files plus single HTTP byte ranges."""

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        file_size = os.path.getsize(path)
        byte_range = _parse_range(range_header, file_size)
        if byte_range is None:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        start, end = byte_range
        content_length = end - start + 1
        file_obj = open(path, "rb")
        file_obj.seek(start)

        self.send_response(206)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
        self.end_headers()
        return _RangeFile(file_obj, content_length)


def _parse_range(header: str, file_size: int) -> tuple[int, int] | None:
    if not header.startswith("bytes=") or file_size <= 0:
        return None
    value = header.removeprefix("bytes=").split(",", 1)[0].strip()
    if "-" not in value:
        return None

    start_text, end_text = value.split("-", 1)
    try:
        if start_text == "":
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return None
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None

    if start < 0 or end < start or start >= file_size:
        return None
    return start, min(end, file_size - 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve this project with byte-range support for video seeking."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RangeRequestHandler)
    print(f"Serving range-enabled GUI at http://{args.host}:{args.port}/gui/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
