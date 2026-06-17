# PRISM Lab Video Analyzer

**Capturing Lost Laboratory Expertise with PRISM and Wearable Cameras**

A Python toolkit that uses Vision Language Models (VLMs) to extract structured
protocol information from laboratory videos, directly addressing the four
challenge tasks from the
[Cultivarium PRISM hackathon](https://prism.cultivarium.org/).

---

## Challenge Tasks

| Video | Lab Procedure | What We Extract |
|-------|--------------|-----------------|
| **A** | Dictyostelium Growth Assay (Protein Quantification) | Number of wells pipetted into a 96-well plate |
| **B** | Antimony Sulfide Nanocrystal Synthesis (Hot Injection) | Exact color changes at each ~30-minute reaction interval |
| **C** | Yeast Transformation Protocol | OD values from spectrophotometer displays (OCR) |
| **D** | LB Agar Plate Preparation | Volume of liquid added (reading graduated glassware) |
| **Stretch** | Any | Full reproducible written protocol synthesised from video + extracted data |

---

## Project Structure

```
.
├── main.py                          # CLI entry point
├── requirements.txt
├── src/
│   ├── video_processor.py           # Frame extraction from video files
│   ├── vlm_client.py                # Claude / Gemini API wrapper
│   └── analyzers/
│       ├── base_analyzer.py         # Abstract base class
│       ├── well_plate_analyzer.py   # Task A – well counting
│       ├── color_change_analyzer.py # Task B – color change detection
│       ├── od_value_analyzer.py     # Task C – OD value OCR
│       ├── volume_analyzer.py       # Task D – liquid volume reading
│       └── protocol_writer.py       # Stretch goal – protocol generation
└── tests/
    └── test_analyzers.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

Create a `.env` file in the project root (or export the variables):

```dotenv
# For Claude (default)
ANTHROPIC_API_KEY=sk-ant-...

# For Gemini
GOOGLE_API_KEY=AIza...
```

---

## Usage

### CLI

```bash
# Video A – count wells pipetted in a 96-well plate
python main.py --video video_a.mp4 --task well_plate

# Video B – detect color changes at 30-minute intervals
python main.py --video video_b.mp4 --task color_change

# Video C – read OD values from spectrophotometer display
python main.py --video video_c.mp4 --task od_values

# Video D – read liquid volume from graduated glassware
python main.py --video video_d.mp4 --task volume

# Stretch goal – write a full protocol
python main.py --video video_a.mp4 --task protocol

# Run all four tasks then synthesise a protocol
python main.py --video video_a.mp4 --task all --output results.json
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--provider` | `claude` or `gemini` | `claude` |
| `--model` | Override the model name | provider default |
| `--output` | Write JSON to file instead of stdout | stdout |
| `--interval` | Frame sampling interval in seconds | task default |
| `--max-frames` | Maximum frames sent to VLM | task default |

### Python API

```python
from src.vlm_client import VLMClient, VLMProvider
from src.analyzers.well_plate_analyzer import WellPlateAnalyzer
from src.analyzers.od_value_analyzer import ODValueAnalyzer
from src.analyzers.protocol_writer import ProtocolWriter

vlm = VLMClient(provider=VLMProvider.CLAUDE)

# Task A
result_a = WellPlateAnalyzer(vlm).analyze("video_a.mp4")
print(f"Wells pipetted: {result_a['final_well_count']}")

# Task C
result_c = ODValueAnalyzer(vlm).analyze("video_c.mp4")
for reading in result_c["all_od_values"]:
    print(f"  {reading['sample_id']}: OD = {reading['od_value']}")

# Stretch – write a protocol incorporating both results
protocol = ProtocolWriter(vlm).analyze(
    "video_a.mp4",
    extracted_results={"well_plate": result_a, "od_values": result_c},
)
print(protocol["protocol_text"])
```

---

## Design Decisions

### VLM-first approach
Rather than relying solely on classical CV (OpenCV contours, colour histograms),
the analyzer sends video frames directly to a VLM.  This handles the specific
challenges called out in the hackathon spec:

- **Object interaction** (pipette entering well) – VLM understands scene context
- **Appearance change** (colour shift over reaction time) – VLM can describe
  subtle hue changes that RGB threshold heuristics miss
- **OCR on instrument displays** (OD values) – modern VLMs read small text in
  natural images reliably
- **Object continuity** (tracking liquid level) – VLM reasons about gradual
  changes across frames

### Structured JSON output
Each analyzer instructs the VLM to return a strict JSON schema with a
`_parse_json_response` fallback to plain text if parsing fails.  The final
result dict always has the same keys so downstream code is predictable.

### Cost awareness
Default `max_frames` caps are set conservatively (20–40 frames) since the
problem statement notes that video analysis can cost a few dollars.  Pass
`--max-frames` to increase coverage when needed.

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests mock both the VLM responses and video I/O so no API keys or real video
files are required.

---

## Resources

- [PRISM main site](https://prism.cultivarium.org/)
- [Cultivarium](https://www.cultivarium.org/)
- Videos and ground-truth protocols: see the workshop GDrive folder
