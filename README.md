# OralExams

Prototype scaffolding for the on-device statistics and regression tutor. The
current repository focuses on the orchestration layer that wires the computer
vision, visual grounding, speech recognition, retrieval-augmented grading and
coaching agents described in the system specification.

* `oral_exams/models.py` – shared dataclasses reflecting the agent bus schema.
* `oral_exams/orchestrator.py` – event loop enforcing confidence gates and routing
  between agents.
* `oral_exams/agents/` – protocol interfaces and default implementations for
  each agent. The repository now ships working OpenCV, Vosk, OCR, retrieval and
  scoring agents so you can exercise the full pipeline locally. Swap any of them
  with production modules (FastVLM, Whisper.cpp, FAISS, OpenELM) as they become
  available.
* `docs/architecture.md` – narrative overview of the architecture, state machine
  and extension points.

## Quickstart (no webcam or microphone required)

Install an editable copy for experimentation:

```bash
pip install -e .
```

Run the canned quickstart script to see how the orchestrator stitches the
agents together:

```bash
python examples/quickstart.py
```

The script wires stub agents that fabricate a vision observation, ASR transcript
and retrieval hit so you can observe the orchestrator state transitions and the
coaching output without any hardware. Replace the inline demo agents with your
own implementations to experiment with different sensing or scoring strategies.

## Live session run

Once you are ready to try a real capture, provide a Hugging Face token (if your
environment requires one for model downloads) and execute the live harness:

```bash
export HF_TOKEN="hf_your_token_here"  # optional, used for Hugging Face downloads
python examples/live_session.py \
  "Interpret R^2 = 0.64 for predicting exam score from hours studied."
```

The script assembles the following on-device agents:

```python
orchestrator = Orchestrator(
    safety=LocalSafetyAgent(config=LocalSafetyConfig(min_mean_luma=args.min_mean_luma)),
    vision=OpenCVVisionAgent(dim_threshold=args.dim_threshold),
    grounding=build_grounding_agent(args.enable_ocr),
    speech=VoskSpeechAgent(hf_token=args.hf_token),
    retrieval=LocalRetrievalAgent(),
    scoring=RuleBasedScoringAgent(),
    coaching=DefaultCoachingAgent(),
)
```

Use the CLI flags to tailor the capture:

* `--duration` – audio recording length in seconds (default: 12).
* `--camera-index` – OpenCV camera index (default: `0`).
* `--min-mean-luma` – minimum average brightness required by the safety agent
  before proceeding (default: `40`). Lower this value if your room lighting is
  intentionally dim.
* `--dim-threshold` – the luma level where the vision agent flags a frame as dim
  for confidence scoring (default: `40`).
* `--enable-ocr` – enable RapidOCR-based math fragment detection (requires the
  optional `rapidocr-onnxruntime` dependency).

Each agent produces tangible signals:

* **Safety/Vision** – uses OpenCV to verify lighting and face count and to track
  a smoothed face ROI.
* **Grounding** – optionally runs RapidOCR to extract math fragments and regions
  from the captured frame.
* **Speech** – records audio with `sounddevice`, transcribes offline with
  Vosk and reports words per minute plus filler counts.
* **Retrieval** – runs TF-IDF retrieval over
  `oral_exams/resources/knowledge_base.json` and computes RAG coverage against
  rubric checkpoints.
* **Scoring & Coaching** – applies rule-based rubric checks, detects the common
  R² misconception, and emits a single actionable coaching step.

### Dependencies

The editable install pulls in the runtime dependencies declared in
`pyproject.toml`, including OpenCV, NumPy, Vosk, RapidOCR (optional), scikit-learn,
sounddevice, and soundfile. If you prefer extras, install the `live` bundle:

```bash
pip install -e .[live]
```

Mac users may need to grant microphone and camera permissions to the Python
interpreter before the first run. On Linux you might need to add your user to the
`audio` group to access the microphone.

