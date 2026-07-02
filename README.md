# Lab Video Review

Local-first lab video review with budgeted uploads to Claude. The tool samples a
small number of frames locally, asks Claude what happened and which moments need
before/after context, then spends the remaining request budget on those focused
windows. Output JSON is compatible with the included browser annotation viewer.

## Workflow

1. **Coarse pass:** sample sparse frames across the full video so Claude can
   build an approximate event timeline and identify low-information ranges.
2. **Focused passes:** Claude requests short windows around meaningful actions,
   measurements, labels, transfers, displays, or other reproducibility-critical
   moments.
3. **Final review:** Claude summarizes what happened, extracts lab actions, and
   produces reproducibility risks, good practices, reproducibility metrics, and
   protocol notes.
4. **Annotated output:** the JSON can be loaded into the browser viewer with the
   source video to show timeline markers and on-video annotations.

The `lab_review` task is Claude-only. Older hackathon analyzers remain in
`src/analyzers/` for reference, but the default CLI path is the budgeted review.

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── src/
│   ├── video_processor.py
│   ├── vlm_client.py
│   └── analyzers/
│       ├── lab_review_analyzer.py
│       └── ... legacy challenge analyzers
├── gui/
│   ├── index.html
│   ├── styles.css
│   └── viewer.js
└── tests/
    └── test_analyzers.py
```

## Setup

```bash
conda env create -p ./.conda -f environment.yml
conda activate ./.conda
pip install -r requirements.txt
```

Create a `.env` file in the project root, or export the variable:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

Run the default budgeted review:

```bash
./.conda/bin/python main.py \
  --video downloads/lab-video.mp4 \
  --output downloads/lab-video.annotations.json \
  --max-claude-requests 6 \
  --max-sampled-frames 70
```

For split recordings, drop the chunk files into `recordings/`. Files are
processed in filename sort order as one virtual timeline, so names like
`001.mp4`, `002.mp4`, ... are preferred.

```bash
./.conda/bin/python main.py \
  --video recordings/ \
  --output downloads/lab-session.annotations.json \
  --preset guided_3usd
```

The `guided_3usd` preset is intended for longer, mostly unedited recordings
where the coarse scan should stay sparse but Claude should spend most of the
budget on guided follow-up windows. It uses:

```text
coarse_interval_seconds = 60
coarse_max_frames = 42
detail_interval_seconds = 4
frames_per_focus = 12
max_focus_windows = 16
max_claude_requests = 18
max_sampled_frames = 240
max_estimated_cost_usd = 3
```

For a 42 minute folder, that means about 42 coarse frames plus up to 192 focused
detail frames before the final synthesis pass. The preflight cost gate aborts
before Claude if the estimated ceiling exceeds `$3.00`.

Before any Claude request is made, the CLI prints a cost estimate and asks:

```text
Continue and run Claude analysis? Type y to continue:
```

Only `y` or `yes` continues. Any other answer aborts before the Claude client is
created and before frames are uploaded.

Tune the sparse sampling strategy:

```bash
./.conda/bin/python main.py \
  --video downloads/lab-video.mp4 \
  --output downloads/lab-video.annotations.json \
  --coarse-interval 45 \
  --coarse-max-frames 16 \
  --detail-interval 3 \
  --frames-per-focus 8 \
  --max-focus-windows 6 \
  --max-claude-requests 6 \
  --max-sampled-frames 64
```

Budget notes:

- `--max-claude-requests` is a hard cap on Claude API calls made by
  `lab_review`, including the first pass and final synthesis.
- `--max-sampled-frames` is a hard cap on uploaded video frames.
- The preflight estimate uses local video dimensions, the configured request
  caps, Claude visual-token accounting, and known API list prices.
- The estimate includes image input tokens, a conservative text-input ceiling,
  and configured output-token ceilings. Taxes and provider-side price changes
  are not included.
- To allow any focused before/after window, use at least `3` requests: first
  pass, one detail pass, final synthesis.
- Cost still depends on model pricing, image size, and output length, so use
  the request and frame caps as practical guardrails before running.

The JSON output includes `request_budget`, `sampling_strategy`,
`event_timeline`, `observed_actions`, `reproducibility_risks`, `thumbs_up`,
`reproducibility_metrics`, and `protocol`.

## Annotation Viewer

Start the range-enabled local static server. The range support matters because
browser video seeking can fail or jump back to the beginning without it.

```bash
./.conda/bin/python scripts/range_server.py --port 8000 > /tmp/prism-gui.log 2>&1 &
echo $! > /tmp/prism-gui.pid
```

Open:

```text
http://127.0.0.1:8000/gui/
```

Use the file controls in the viewer to load your video and the generated
`*.annotations.json` file.

Stop the server when done:

```bash
kill "$(cat /tmp/prism-gui.pid)"
rm /tmp/prism-gui.pid
```

## Tests

```bash
python -m unittest
```
