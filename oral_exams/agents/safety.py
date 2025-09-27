"""Safety and privacy guardrails."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import SafetyResult, SafetySignal, TurnBundle


@dataclass
class SafetyConfig:
    enforce_offline: bool = True
    max_buffer_seconds: int = 10


class SafetyAgent(Protocol):
    def check(self, bundle: TurnBundle) -> SafetyResult:
        ...


class DefaultSafetyAgent:
    """Placeholder safety implementation enforcing the global contracts."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def check(self, bundle: TurnBundle) -> SafetyResult:
        # Real implementation would inspect device state and bundle metadata.
        return SafetyResult(signal=SafetySignal.SAFE_OK)

