"""Unit tests for PRISM Lab Video Analyzer.

These tests use mocks so no actual video files or API keys are required.
"""

import base64
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_b64() -> str:
    """Return a tiny valid base64-encoded JPEG string for testing."""
    # 1×1 white JPEG (minimal valid JPEG bytes)
    tiny_jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
        b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
        b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br"
        b"\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJST"
        b"UVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93"
        b"\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa"
        b"\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8"
        b"\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5"
        b"\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0e\xff\xd9"
    )
    return base64.b64encode(tiny_jpeg).decode()


SAMPLE_FRAME = {
    "timestamp_sec": 0.0,
    "frame_index": 0,
    "image_b64": _make_jpeg_b64(),
}


# ---------------------------------------------------------------------------
# VLMClient tests
# ---------------------------------------------------------------------------

class TestVLMClientInit(unittest.TestCase):
    def test_missing_claude_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove key if present
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            from src.vlm_client import VLMClient, VLMProvider
            with self.assertRaises(EnvironmentError):
                VLMClient(provider=VLMProvider.CLAUDE)

    def test_missing_gemini_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("GOOGLE_API_KEY", None)
            from src.vlm_client import VLMClient, VLMProvider
            with self.assertRaises(EnvironmentError):
                VLMClient(provider=VLMProvider.GEMINI)

    def test_ollama_init_needs_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            from src.vlm_client import VLMClient, VLMProvider

            client = VLMClient(provider=VLMProvider.OLLAMA)

        self.assertEqual(client.provider, VLMProvider.OLLAMA)
        self.assertEqual(client.model, "qwen2.5vl:7b")

    def test_ollama_analyze_frames(self):
        from src.vlm_client import VLMClient, VLMProvider

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b'{"response": "{\\"ok\\": true}"}'

        with patch("urllib.request.urlopen", return_value=FakeResponse()) as mock_urlopen:
            client = VLMClient(provider=VLMProvider.OLLAMA)
            result = client.analyze_frames([SAMPLE_FRAME], "Respond using JSON")

        self.assertEqual(result, '{"ok": true}')
        request = mock_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")

    def test_claude_init_with_key(self):
        """Claude client initializes when API key is set."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            fake_anthropic = types.SimpleNamespace(Anthropic=MagicMock())
            with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
                with patch("anthropic.Anthropic") as mock_cls:
                    mock_cls.return_value = MagicMock()
                    from importlib import reload
                    import src.vlm_client as vc
                    reload(vc)
                    client = vc.VLMClient(provider=vc.VLMProvider.CLAUDE)
                    self.assertEqual(client.provider, vc.VLMProvider.CLAUDE)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

class TestJSONParsing(unittest.TestCase):
    def test_well_plate_parse_valid_json(self):
        from src.analyzers.well_plate_analyzer import _parse_json_response
        payload = {
            "frames": [{"timestamp_sec": 0, "pipetting_visible": True,
                        "wells_identified": ["A1"], "running_total": 1}],
            "final_well_count": 1,
            "notes": "",
        }
        result = _parse_json_response(json.dumps(payload))
        self.assertEqual(result["final_well_count"], 1)
        self.assertEqual(len(result["frames"]), 1)

    def test_well_plate_parse_invalid_returns_defaults(self):
        from src.analyzers.well_plate_analyzer import _parse_json_response
        result = _parse_json_response("not json at all")
        self.assertIsNone(result["final_well_count"])
        self.assertEqual(result["frames"], [])

    def test_color_change_parse_valid_json(self):
        from src.analyzers.color_change_analyzer import _parse_json_response
        payload = {
            "frames": [],
            "color_change_summary": "Yellow to red",
            "notable_transitions": [],
        }
        result = _parse_json_response(json.dumps(payload))
        self.assertEqual(result["color_change_summary"], "Yellow to red")

    def test_od_value_parse_valid_json(self):
        from src.analyzers.od_value_analyzer import _parse_json_response
        payload = {
            "frames": [],
            "all_od_values": [{"sample_id": "S1", "od_value": 0.45,
                                "wavelength_nm": 600}],
            "notes": "",
        }
        result = _parse_json_response(json.dumps(payload))
        self.assertEqual(result["all_od_values"][0]["od_value"], 0.45)

    def test_volume_parse_valid_json(self):
        from src.analyzers.volume_analyzer import _parse_json_response
        payload = {
            "frames": [],
            "volume_additions": [{"timestamp_sec": 30, "volume_ml": 250,
                                   "description": "Added buffer"}],
            "total_volume_added_ml": 250,
            "summary": "250 mL added",
        }
        result = _parse_json_response(json.dumps(payload))
        self.assertEqual(result["total_volume_added_ml"], 250)


# ---------------------------------------------------------------------------
# Analyzer tests (mocked VLM + video)
# ---------------------------------------------------------------------------

def _make_mock_vlm(response_json: dict) -> MagicMock:
    vlm = MagicMock()
    vlm.analyze_frames.return_value = json.dumps(response_json)
    return vlm


def _patch_extract_frames(frames, module: str = "src.analyzers.well_plate_analyzer"):
    """Patch extract_frames in the given analyzer module."""
    return patch(f"{module}.extract_frames", return_value=frames)


def _patch_extract_frames_at_timestamps(frames):
    return patch(
        "src.analyzers.color_change_analyzer.extract_frames_at_timestamps",
        return_value=frames,
    )


def _patch_get_metadata(duration=60.0):
    return patch(
        "src.analyzers.color_change_analyzer.get_video_metadata",
        return_value={"fps": 30, "total_frames": 1800,
                      "duration_sec": duration, "width": 1280, "height": 720},
    )


def _patch_lab_review_metadata(duration=120.0):
    return patch(
        "src.analyzers.lab_review_analyzer.get_video_metadata",
        return_value={"fps": 30, "total_frames": int(duration * 30),
                      "duration_sec": duration, "width": 1280, "height": 720},
    )


class TestWellPlateAnalyzer(unittest.TestCase):
    def test_analyze_returns_well_count(self):
        from src.analyzers.well_plate_analyzer import WellPlateAnalyzer

        response = {
            "frames": [{"timestamp_sec": 0, "pipetting_visible": True,
                        "wells_identified": ["A1", "A2"], "running_total": 2}],
            "final_well_count": 8,
            "notes": "Clear pipetting visible",
        }
        vlm = _make_mock_vlm(response)

        with _patch_extract_frames([SAMPLE_FRAME], "src.analyzers.well_plate_analyzer"):
            result = WellPlateAnalyzer(vlm).analyze("fake.mp4")

        self.assertEqual(result["final_well_count"], 8)
        self.assertEqual(result["task"], "well_plate_counting")
        self.assertIn("frame_details", result)
        vlm.analyze_frames.assert_called_once()

    def test_analyze_handles_bad_json(self):
        from src.analyzers.well_plate_analyzer import WellPlateAnalyzer

        vlm = MagicMock()
        vlm.analyze_frames.return_value = "The model returned plain text."

        with _patch_extract_frames([SAMPLE_FRAME], "src.analyzers.well_plate_analyzer"):
            result = WellPlateAnalyzer(vlm).analyze("fake.mp4")

        self.assertIsNone(result["final_well_count"])


class TestColorChangeAnalyzer(unittest.TestCase):
    def test_analyze_returns_summary(self):
        from src.analyzers.color_change_analyzer import ColorChangeAnalyzer

        response = {
            "frames": [{"timestamp_sec": 0, "color_description": "yellow",
                        "change_from_previous": "no change", "hex_approximate": "#FFFF00"}],
            "color_change_summary": "Solution changed from yellow to dark brown.",
            "notable_transitions": [
                {"from_timestamp_sec": 0, "to_timestamp_sec": 30,
                 "description": "yellow → orange"}
            ],
        }
        vlm = _make_mock_vlm(response)

        with _patch_get_metadata(120.0), _patch_extract_frames_at_timestamps([SAMPLE_FRAME]):
            result = ColorChangeAnalyzer(vlm).analyze("fake.mp4", interval_minutes=30.0)

        self.assertIn("color_change_summary", result)
        self.assertEqual(result["task"], "color_change_detection")
        self.assertEqual(len(result["notable_transitions"]), 1)


class TestODValueAnalyzer(unittest.TestCase):
    def test_analyze_returns_od_values(self):
        from src.analyzers.od_value_analyzer import ODValueAnalyzer

        response = {
            "frames": [],
            "all_od_values": [
                {"sample_id": "blank", "od_value": 0.0, "wavelength_nm": 600},
                {"sample_id": "S1", "od_value": 0.312, "wavelength_nm": 600},
            ],
            "notes": "Display clearly visible",
        }
        vlm = _make_mock_vlm(response)

        with _patch_extract_frames([SAMPLE_FRAME], "src.analyzers.od_value_analyzer"):
            result = ODValueAnalyzer(vlm).analyze("fake.mp4")

        self.assertEqual(result["task"], "od_value_reading")
        self.assertEqual(len(result["all_od_values"]), 2)
        self.assertAlmostEqual(result["all_od_values"][1]["od_value"], 0.312)


class TestYeastTransformationAnalyzer(unittest.TestCase):
    def test_analyze_returns_protocol_demo(self):
        from src.analyzers.yeast_transformation_analyzer import (
            YeastTransformationAnalyzer,
        )

        response = {
            "od_values": [
                {"sample_id": "culture", "od_value": 0.62, "wavelength_nm": 600}
            ],
            "observed_actions": [
                {
                    "timestamp_sec": 12,
                    "action": "Measure yeast culture OD",
                    "materials": ["spectrophotometer", "cuvette"],
                    "measurement": "OD600 0.62",
                    "confidence": "high",
                }
            ],
            "reproducibility_risks": [
                {
                    "timestamp_sec": 12,
                    "action": "Measure yeast culture OD",
                    "issue": "Sample ID is not visible in the frame.",
                    "severity": "High",
                    "suggested_fix": "Record the sample ID alongside the OD reading.",
                    "confidence": "medium",
                }
            ],
            "thumbs_up": [
                {
                    "timestamp_sec": 12,
                    "practice": "Culture density is measured before transformation.",
                    "why_it_helps": "OD600 anchors the culture state for reproduction.",
                    "confidence": "high",
                }
            ],
            "protocol": {
                "title": "Yeast Transformation Protocol",
                "materials": ["yeast culture", "spectrophotometer"],
                "steps": ["Measure yeast culture density at OD600."],
                "uncertainties": ["Transformation reagent volumes not visible."],
            },
            "notes": "Display is readable.",
        }
        vlm = _make_mock_vlm(response)

        with _patch_extract_frames(
            [SAMPLE_FRAME], "src.analyzers.yeast_transformation_analyzer"
        ):
            result = YeastTransformationAnalyzer(vlm).analyze("fake.mp4")

        self.assertEqual(result["task"], "yeast_transformation_protocol")
        self.assertEqual(result["od_values"][0]["od_value"], 0.62)
        self.assertEqual(result["reproducibility_risks"][0]["severity"], "High")
        self.assertIn("OD600", result["thumbs_up"][0]["why_it_helps"])
        self.assertIn("Measure yeast", result["protocol"]["steps"][0])

    def test_bad_json_returns_uncertainty(self):
        from src.analyzers.yeast_transformation_analyzer import _parse_json_response

        result = _parse_json_response("plain text")

        self.assertEqual(result["od_values"], [])
        self.assertEqual(result["reproducibility_risks"], [])
        self.assertEqual(result["thumbs_up"], [])
        self.assertIn("Could not parse", result["protocol"]["uncertainties"][0])


class TestLabReviewAnalyzer(unittest.TestCase):
    def test_budgeted_review_uses_requested_detail_window(self):
        from src.analyzers.lab_review_analyzer import LabReviewAnalyzer

        first_pass = {
            "video_summary": "A transfer appears to happen around 30 seconds.",
            "event_timeline": [
                {
                    "timestamp_sec": 30,
                    "event": "Scientist handles a tube.",
                    "evidence": "Tube and rack visible.",
                    "confidence": "medium",
                }
            ],
            "focus_requests": [
                {
                    "start_sec": 24,
                    "end_sec": 36,
                    "reason": "Need before/after shots for the transfer.",
                    "priority": "high",
                }
            ],
            "low_information_ranges": [],
            "notes": "",
        }
        detail = {
            "observed_actions": [
                {
                    "timestamp_sec": 30,
                    "action": "Transfer liquid into a tube",
                    "materials": ["tube", "pipette"],
                    "measurement": None,
                    "confidence": "medium",
                }
            ],
            "reproducibility_risks": [
                {
                    "timestamp_sec": 30,
                    "action": "Transfer liquid",
                    "issue": "Volume is not visible.",
                    "severity": "High",
                    "suggested_fix": "Record transfer volume.",
                    "confidence": "medium",
                }
            ],
            "thumbs_up": [],
            "focus_requests": [],
            "notes": "",
        }
        final = {
            "review_summary": "The video shows a tube transfer around 30 seconds.",
            "event_timeline": first_pass["event_timeline"],
            "observed_actions": detail["observed_actions"],
            "reproducibility_risks": detail["reproducibility_risks"],
            "thumbs_up": [],
            "reproducibility_metrics": [
                {
                    "metric": "critical-parameters",
                    "score": 2,
                    "evidence": "Transfer volume is not visible.",
                    "recommendation": "Capture labels and volumes.",
                }
            ],
            "protocol": {
                "title": "Lab Video Review",
                "materials": ["tube", "pipette"],
                "steps": ["Transfer liquid into a tube."],
                "uncertainties": ["Transfer volume is missing."],
            },
            "notes": "",
        }

        vlm = MagicMock()
        vlm.provider = "claude"
        vlm.analyze_frames.side_effect = [
            json.dumps(first_pass),
            json.dumps(detail),
            json.dumps(final),
        ]

        with (
            _patch_lab_review_metadata(120.0),
            patch(
                "src.analyzers.lab_review_analyzer.extract_frames",
                return_value=[SAMPLE_FRAME],
            ),
            patch(
                "src.analyzers.lab_review_analyzer.extract_frames_at_timestamps",
                return_value=[{**SAMPLE_FRAME, "timestamp_sec": 24.0}],
            ),
        ):
            result = LabReviewAnalyzer(vlm).analyze(
                "fake.mp4",
                max_claude_requests=3,
                max_sampled_frames=5,
            )

        self.assertEqual(result["task"], "lab_review")
        self.assertEqual(result["request_budget"]["used_claude_requests"], 3)
        self.assertEqual(vlm.analyze_frames.call_count, 3)
        self.assertEqual(result["observed_actions"][0]["action"],
                         "Transfer liquid into a tube")
        self.assertEqual(result["reproducibility_risks"][0]["severity"], "High")
        self.assertEqual(result["reproducibility_metrics"][0]["score"], 2)

    def test_claude_only_enforced(self):
        from src.analyzers.lab_review_analyzer import LabReviewAnalyzer

        vlm = MagicMock()
        vlm.provider = "ollama"

        with self.assertRaises(ValueError):
            LabReviewAnalyzer(vlm).analyze("fake.mp4")


class TestVolumeAnalyzer(unittest.TestCase):
    def test_analyze_returns_volume(self):
        from src.analyzers.volume_analyzer import VolumeAnalyzer

        response = {
            "frames": [],
            "volume_additions": [
                {"timestamp_sec": 15, "volume_ml": 500,
                 "description": "Added 500 mL LB broth"}
            ],
            "total_volume_added_ml": 500,
            "summary": "500 mL of LB broth added to bottle.",
        }
        vlm = _make_mock_vlm(response)

        with _patch_extract_frames([SAMPLE_FRAME], "src.analyzers.volume_analyzer"):
            result = VolumeAnalyzer(vlm).analyze("fake.mp4")

        self.assertEqual(result["task"], "volume_reading")
        self.assertEqual(result["total_volume_added_ml"], 500)
        self.assertEqual(len(result["volume_additions"]), 1)


class TestProtocolWriter(unittest.TestCase):
    def test_protocol_returns_text(self):
        from src.analyzers.protocol_writer import ProtocolWriter

        protocol_text = "# LB Agar Protocol\n\n## Materials\n- LB Broth 500 mL\n"
        vlm = MagicMock()
        vlm.analyze_frames.return_value = protocol_text

        with _patch_extract_frames([SAMPLE_FRAME], "src.analyzers.protocol_writer"):
            result = ProtocolWriter(vlm).analyze("fake.mp4")

        self.assertEqual(result["task"], "protocol_writing")
        self.assertEqual(result["protocol_text"], protocol_text)

    def test_protocol_with_extracted_data(self):
        from src.analyzers.protocol_writer import ProtocolWriter

        vlm = MagicMock()
        vlm.analyze_frames.return_value = "Full protocol text"

        extracted = {"volume_reading": {"total_volume_added_ml": 250}}

        with _patch_extract_frames([SAMPLE_FRAME], "src.analyzers.protocol_writer"):
            result = ProtocolWriter(vlm).analyze(
                "fake.mp4", extracted_results=extracted
            )

        self.assertEqual(result["protocol_text"], "Full protocol text")
        # Verify that extracted data was included in the prompt
        call_args = vlm.analyze_frames.call_args
        prompt = call_args[0][1]
        self.assertIn("250", prompt)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCLI(unittest.TestCase):
    def test_missing_video_exits_nonzero(self):
        from main import main
        ret = main(["--video", "/nonexistent/path.mp4", "--task", "well_plate"])
        self.assertEqual(ret, 1)

    def test_missing_api_key_exits_nonzero(self):
        import os, tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            from main import main
            ret = main(["--video", video_path, "--task", "well_plate",
                        "--provider", "claude"])
            self.assertEqual(ret, 1)
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_help_exits_zero(self):
        from main import build_parser
        parser = build_parser()
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_lab_review_cost_prompt_no_stops_before_client(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            from main import main

            estimate = {
                "model": "claude-opus-4-5",
                "pricing": {"input": 5.0, "output": 25.0},
                "video": {"width": 1280, "height": 720, "duration_sec": 60},
                "resized_frame": {
                    "width": 1024,
                    "height": 576,
                    "visual_tokens": 777,
                },
                "max_uploaded_frames": 10,
                "max_claude_requests": 3,
                "image_input_tokens": 7770,
                "output_token_ceiling": 10240,
                "image_input_cost_usd": 0.03885,
                "output_cost_ceiling_usd": 0.256,
                "estimated_ceiling_usd": 0.29485,
                "notes": "test",
            }

            with (
                patch("main.estimate_lab_review_cost", return_value=estimate),
                patch("builtins.input", return_value="n"),
                patch("src.vlm_client.VLMClient") as mock_client,
            ):
                ret = main(["--video", video_path])

            self.assertEqual(ret, 1)
            mock_client.assert_not_called()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_lab_review_cost_prompt_yes_runs_task(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            from main import main

            estimate = {
                "model": "claude-opus-4-5",
                "pricing": {"input": 5.0, "output": 25.0},
                "video": {"width": 1280, "height": 720, "duration_sec": 60},
                "resized_frame": {
                    "width": 1024,
                    "height": 576,
                    "visual_tokens": 777,
                },
                "max_uploaded_frames": 10,
                "max_claude_requests": 3,
                "image_input_tokens": 7770,
                "output_token_ceiling": 10240,
                "image_input_cost_usd": 0.03885,
                "output_cost_ceiling_usd": 0.256,
                "estimated_ceiling_usd": 0.29485,
                "notes": "test",
            }

            with (
                patch("main.estimate_lab_review_cost", return_value=estimate),
                patch("builtins.input", return_value="y"),
                patch("src.vlm_client.VLMClient", return_value=MagicMock()),
                patch("main.run_task", return_value={"task": "lab_review"}) as mock_run_task,
            ):
                ret = main(["--video", video_path])

            self.assertEqual(ret, 0)
            mock_run_task.assert_called_once()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_cost_estimator_handles_portrait_video(self):
        from main import estimate_lab_review_cost

        with patch(
            "src.video_processor.get_video_metadata",
            return_value={
                "fps": 30,
                "total_frames": 1800,
                "duration_sec": 60,
                "width": 1080,
                "height": 1920,
            },
        ):
            cost = estimate_lab_review_cost(
                "portrait.mp4",
                "claude-opus-4-5",
                {"max_sampled_frames": 70},
            )

        self.assertEqual(cost["resized_frame"]["width"], 1024)
        self.assertGreater(cost["resized_frame"]["height"], 1700)
        self.assertEqual(cost["max_uploaded_frames"], 50)
        self.assertGreater(cost["estimated_ceiling_usd"], 0)


if __name__ == "__main__":
    unittest.main()
