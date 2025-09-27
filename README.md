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

## Development

Install an editable copy for experimentation:

```bash
pip install -e .
```

Then run a live turn using your webcam and microphone. The command below
captures a frame from camera index `0`, records 12 seconds of audio, downloads
the Vosk speech model (using your Hugging Face token if required), performs OCR
grounding when available, and prints the orchestrator decision.

```bash
export HF_TOKEN="hf_your_token_here"  # optional, used for Hugging Face downloads
python examples/live_session.py \
  "Interpret R^2 = 0.64 for predicting exam score from hours studied."
```

You can customise the run by passing `--duration`, `--camera-index`, or
`--enable-ocr` to turn on RapidOCR-based math fragment detection (requires the
`rapidocr-onnxruntime` package).

The script assembles the following real agents:

```python
orchestrator = Orchestrator(
    safety=LocalSafetyAgent(),
    vision=OpenCVVisionAgent(),
    grounding=build_grounding_agent(args.enable_ocr),
    speech=VoskSpeechAgent(hf_token=args.hf_token),
    retrieval=LocalRetrievalAgent(),
    scoring=RuleBasedScoringAgent(),
    coaching=DefaultCoachingAgent(),
)
```

Each agent produces tangible signals:

* **Safety/Vision** – uses OpenCV to verify lighting and face count and to track
  a smoothed face ROI.
* **Grounding** – optionally runs RapidOCR to extract math fragments and regions
  from the captured frame.
* **Speech** – records audio with `sounddevice`, transcribes offline with
  Vosk and reports words per minute plus filler counts.
* **Retrieval** – runs TF-IDF retrieval over `oral_exams/resources/knowledge_base.json`
  and computes RAG coverage against rubric checkpoints.
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

