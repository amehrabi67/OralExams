# OralExams

Prototype scaffolding for the on-device statistics and regression tutor. The
current repository focuses on the orchestration layer that wires the computer
vision, visual grounding, speech recognition, retrieval-augmented grading and
coaching agents described in the system specification.

* `oral_exams/models.py` – shared dataclasses reflecting the agent bus schema.
* `oral_exams/orchestrator.py` – event loop enforcing confidence gates and routing
  between agents.
* `oral_exams/agents/` – protocol interfaces and stub implementations for each
  agent. Swap these with production modules (OpenCV, FastVLM, Whisper/Vosk,
  FAISS, OpenELM) to deliver the full experience.
* `docs/architecture.md` – narrative overview of the architecture, state machine
  and extension points.

## Development

Install an editable copy for experimentation:

```bash
pip install -e .
```

Then build concrete agent implementations and inject them into the
`Orchestrator` to run end-to-end simulations or hardware-in-the-loop tests.

