"""Speech transcription agent wrapper."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from statistics import mean
from typing import Protocol

import numpy as np
import requests
import soundfile as sf

from ..models import SpeechObservation, TurnBundle


class SpeechAgent(Protocol):
    def transcribe(self, bundle: TurnBundle) -> SpeechObservation:
        ...


class VoskSpeechAgent:
    """Offline speech transcription using the Vosk small English model."""

    def __init__(
        self,
        model_name: str = "vosk-model-small-en-us-0.15",
        cache_dir: str | Path | None = None,
        hf_token: str | None = None,
    ) -> None:
        # Import vosk lazily so the module can be imported in environments
        # where vosk isn't installed. If vosk is missing, constructing this
        # agent will raise ImportError.
        try:
            from vosk import Model  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError(
                "vosk is required for VoskSpeechAgent but is not installed"
            ) from exc

        model_path = _ensure_vosk_model(model_name, cache_dir, hf_token)
        self._model = Model(str(model_path))

    def transcribe(self, bundle: TurnBundle) -> SpeechObservation:
        if bundle.raw_audio_path is None:
            raise ValueError("TurnBundle.raw_audio_path must be provided for transcription.")

        audio_path = Path(bundle.raw_audio_path)
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype != np.int16:
            data = np.clip(data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
        sf.write(audio_path, data, sr, subtype="PCM_16")  # ensure mono PCM16
        # Import KaldiRecognizer lazily to avoid module-level import errors
        try:
            from vosk import KaldiRecognizer  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError(
                "vosk is required for VoskSpeechAgent but is not installed"
            ) from exc

        recognizer = KaldiRecognizer(self._model, sr)
        recognizer.SetWords(True)

        confidences: list[float] = []
        segments: list[str] = []

        with sf.SoundFile(audio_path, "rb") as audio_file:
            while True:
                chunk = audio_file.buffer_read(4000, dtype="int16")
                if not chunk:
                    break
                if recognizer.AcceptWaveform(chunk):
                    result_blob = recognizer.Result()
                    confidences.extend(_word_confidences(result_blob))
                    segments.append(_extract_text(result_blob))
        final_blob = recognizer.FinalResult()
        confidences.extend(_word_confidences(final_blob))
        segments.append(_extract_text(final_blob))

        text = " ".join(seg for seg in segments if seg).strip()
        words = text.split()
        duration_seconds = len(data) / sr if sr else 0.0
        wpm = (len(words) / duration_seconds) * 60 if duration_seconds else 0.0
        filler_words = {"uh", "um", "like", "er", "ah"}
        fillers = sum(1 for word in words if word.lower() in filler_words)

        asr_conf = mean(confidences) if confidences else 0.5
        return SpeechObservation(
            transcript=text,
            asr_conf=float(min(max(asr_conf, 0.0), 1.0)),
            wpm=float(wpm),
            fillers_per_min=float(fillers / (duration_seconds / 60)) if duration_seconds else 0.0,
        )


def _ensure_vosk_model(
    model_name: str, cache_dir: str | Path | None, hf_token: str | None
) -> Path:
    base_dir = Path(cache_dir or Path.home() / ".cache" / "oral_exams" / "vosk")
    base_dir.mkdir(parents=True, exist_ok=True)
    target_dir = base_dir / model_name
    if (target_dir / "am" / "final.mdl").exists():
        return target_dir

    zip_path = base_dir / f"{model_name}.zip"
    if not zip_path.exists():
        # Try official Vosk website first
        url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(zip_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1_048_576):
                        if chunk:
                            handle.write(chunk)
        except requests.exceptions.RequestException:
            # Fallback to Hugging Face
            url = f"https://huggingface.co/alphacep/{model_name}/resolve/main/{model_name}.zip"
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            with requests.get(url, headers=headers, stream=True, timeout=300) as response:
                response.raise_for_status()
                with open(zip_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1_048_576):
                        if chunk:
                            handle.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(base_dir)

    return target_dir


def _word_confidences(result_blob: str) -> list[float]:
    confidences: list[float] = []
    if not result_blob:
        return confidences
    try:
        payload = json.loads(result_blob)
    except json.JSONDecodeError:
        return confidences

    for word in payload.get("result", []) or []:
        conf = float(word.get("conf", 0.0))
        if conf:
            confidences.append(conf)
    return confidences


def _extract_text(result_blob: str) -> str:
    if not result_blob:
        return ""
    try:
        payload = json.loads(result_blob)
    except json.JSONDecodeError:
        return ""
    return (payload.get("text") or "").strip()

