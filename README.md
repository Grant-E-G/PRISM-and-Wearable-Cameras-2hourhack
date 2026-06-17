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
| **C demo** | Yeast Transformation Protocol | OD values, observed actions, reproducibility risks, good practices, and a compact protocol draft |
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
│   ├── vlm_client.py                # Claude / Gemini / Ollama API wrapper
│   └── analyzers/
│       ├── base_analyzer.py         # Abstract base class
│       ├── well_plate_analyzer.py   # Task A – well counting
│       ├── color_change_analyzer.py # Task B – color change detection
│       ├── od_value_analyzer.py     # Task C – OD value OCR
│       ├── yeast_transformation_analyzer.py # Task C demo – protocol capture
│       ├── volume_analyzer.py       # Task D – liquid volume reading
│       └── protocol_writer.py       # Stretch goal – protocol generation
├── gui/
│   ├── index.html                   # Browser annotation viewer
│   ├── styles.css
│   └── viewer.js
└── tests/
    └── test_analyzers.py
```

---

## Setup

### 1. Create the conda environment

```bash
conda env create -p ./.conda -f environment.yml
conda activate ./.conda
```

### 2. Install dependencies with pip only

```bash
pip install -r requirements.txt
```

### 3. Set API keys

Create a `.env` file in the project root (or export the variables):

```dotenv
# For Claude (default)
ANTHROPIC_API_KEY=sk-ant-...

# For Gemini
GOOGLE_API_KEY=AIza...
```

Claude and Gemini are optional if you run locally with Ollama:

```bash
ollama pull qwen2.5vl:7b
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

# Video C demo – extract a yeast transformation protocol draft locally
python main.py --video video_c.mp4 --task yeast_protocol --provider ollama

# Video D – read liquid volume from graduated glassware
python main.py --video video_d.mp4 --task volume

# Stretch goal – write a full protocol
python main.py --video video_a.mp4 --task protocol

# Run all four tasks then synthesise a protocol
python main.py --video video_a.mp4 --task all --output results.json
```

### Annotation Viewer

The GUI is a static browser viewer. It plays the lab video at the top, overlays
active annotations on the video, shows full annotation text below, and lets you
jump by clicking timeline markers or annotation cards.

The current local demo defaults to:

- Video: `downloads/yeast_protocol_1min_gui.mp4`
- JSON: `downloads/yeast_protocol_1min.annotations.json`

Both files live in `downloads/`, which is ignored by git.

#### Generate Annotations

Run the analyzer and write JSON for the viewer:

```bash
./.conda/bin/python main.py \
  --video downloads/yeast_protocol_1min.mp4 \
  --task yeast_protocol \
  --provider claude \
  --interval 5 \
  --max-frames 12 \
  --output downloads/yeast_protocol_1min.annotations.json
```

For a cheaper quick check, use fewer frames:

```bash
./.conda/bin/python main.py \
  --video downloads/yeast_protocol_1min.mp4 \
  --task yeast_protocol \
  --provider claude \
  --interval 10 \
  --max-frames 6 \
  --output downloads/yeast_protocol_1min.annotations.json
```

#### Optimize Video For The Browser

If the GUI video seeking feels slow or jumps back to the beginning, create a
smaller browser-facing copy:

```bash
ffmpeg -y \
  -i downloads/yeast_protocol_1min.mp4 \
  -map 0:v:0 \
  -map 0:a? \
  -map_metadata -1 \
  -write_tmcd 0 \
  -vf "scale='min(1280,iw)':-2" \
  -c:v libx264 \
  -preset veryfast \
  -crf 23 \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  downloads/yeast_protocol_1min_gui.mp4
```

#### Start The Web Server

```bash
./.conda/bin/python -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/gui/
```

If your browser cached old assets, add a query string:

```text
http://127.0.0.1:8000/gui/?reload=1
```

#### Stop The Web Server

If the server is running in the foreground, press `Ctrl-C` in that terminal.

If you started it in the background, save the PID:

```bash
./.conda/bin/python -m http.server 8000 > /tmp/prism-gui.log 2>&1 &
echo $! > /tmp/prism-gui.pid
```

Stop it later with:

```bash
kill "$(cat /tmp/prism-gui.pid)"
rm /tmp/prism-gui.pid
```

Check whether port `8000` is already serving the GUI:

```bash
curl -I http://127.0.0.1:8000/gui/
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--provider` | `claude`, `gemini`, or `ollama` | `claude` |
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

# For local no-key runs
vlm = VLMClient(provider=VLMProvider.OLLAMA)

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

The yeast protocol demo also emits `reproducibility_risks` with severity values
of `Very High`, `High`, `Medium`, or `Low`, plus `thumbs_up` annotations for
extra good practices observed in the footage.

### Cost awareness
Default `max_frames` caps are set conservatively (20–40 frames) since the
problem statement notes that video analysis can cost a few dollars.  Pass
`--max-frames` to increase coverage when needed.

---

## Running Tests

```bash
./.conda/bin/python -m pytest tests/ -v
```

Tests mock both the VLM responses and video I/O so no API keys or real video
files are required.

---

## Resources

- [PRISM main site](https://prism.cultivarium.org/)
- [Cultivarium](https://www.cultivarium.org/)
- Videos and ground-truth protocols: see the workshop GDrive folder
