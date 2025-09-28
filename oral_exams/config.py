"""Configuration system for the tutoring system."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class SafetyConfig:
    """Safety agent configuration."""
    min_mean_luma: float = 40.0
    max_faces: int = 1
    face_detector: Optional[str] = None
    enforce_offline: bool = True
    max_buffer_seconds: int = 10


@dataclass
class VisionConfig:
    """Vision agent configuration."""
    min_cv_conf: float = 0.60
    roi_detection_enabled: bool = True
    face_detection_enabled: bool = True
    board_detection_enabled: bool = True


@dataclass
class GroundingConfig:
    """Grounding agent configuration."""
    min_vlm_conf: float = 0.60
    ocr_enabled: bool = False
    math_detection_enabled: bool = True


@dataclass
class SpeechConfig:
    """Speech agent configuration."""
    model_name: str = "vosk-model-small-en-us-0.15"
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None
    sample_rate: int = 16000
    duration: float = 12.0


@dataclass
class RetrievalConfig:
    """Retrieval agent configuration."""
    min_rag_cov: float = 0.70
    knowledge_base_path: str = "oral_exams/resources/knowledge_base.json"
    max_hits: int = 5


@dataclass
class ScoringConfig:
    """Scoring agent configuration."""
    content_weight: float = 0.7
    voice_weight: float = 0.3
    misconception_penalty: float = 2.0
    concept_bonus: float = 1.0


@dataclass
class CoachingConfig:
    """Coaching agent configuration."""
    feedback_detailed: bool = True
    adaptive_strategies: bool = True
    personalized_feedback: bool = True


@dataclass
class TutoringConfig:
    """Main tutoring system configuration."""
    safety: SafetyConfig
    vision: VisionConfig
    grounding: GroundingConfig
    speech: SpeechConfig
    retrieval: RetrievalConfig
    scoring: ScoringConfig
    coaching: CoachingConfig
    
    # Global settings
    debug_mode: bool = False
    log_level: str = "INFO"
    output_dir: str = "./tutoring_output"
    
    @classmethod
    def default(cls) -> 'TutoringConfig':
        """Create default configuration."""
        return cls(
            safety=SafetyConfig(),
            vision=VisionConfig(),
            grounding=GroundingConfig(),
            speech=SpeechConfig(),
            retrieval=RetrievalConfig(),
            scoring=ScoringConfig(),
            coaching=CoachingConfig(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TutoringConfig':
        """Create from dictionary."""
        return cls(
            safety=SafetyConfig(**data.get('safety', {})),
            vision=VisionConfig(**data.get('vision', {})),
            grounding=GroundingConfig(**data.get('grounding', {})),
            speech=SpeechConfig(**data.get('speech', {})),
            retrieval=RetrievalConfig(**data.get('retrieval', {})),
            scoring=ScoringConfig(**data.get('scoring', {})),
            coaching=CoachingConfig(**data.get('coaching', {})),
            debug_mode=data.get('debug_mode', False),
            log_level=data.get('log_level', 'INFO'),
            output_dir=data.get('output_dir', './tutoring_output'),
        )
    
    def save(self, file_path: Path) -> None:
        """Save configuration to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: Path) -> 'TutoringConfig':
        """Load configuration from file."""
        if not file_path.exists():
            config = cls.default()
            config.save(file_path)
            return config
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Try to update nested configs
                for config_name in ['safety', 'vision', 'grounding', 'speech', 'retrieval', 'scoring', 'coaching']:
                    config = getattr(self, config_name)
                    if hasattr(config, key):
                        setattr(config, key, value)
                        break


class ConfigManager:
    """Configuration manager for the tutoring system."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("tutoring_config.json")
        self.config = TutoringConfig.load(self.config_file)
    
    def get_config(self) -> TutoringConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration."""
        self.config.update(**kwargs)
        self.save_config()
    
    def save_config(self) -> None:
        """Save configuration to file."""
        self.config.save(self.config_file)
    
    def reset_to_default(self) -> None:
        """Reset configuration to defaults."""
        self.config = TutoringConfig.default()
        self.save_config()
    
    def get_safety_config(self) -> SafetyConfig:
        """Get safety configuration."""
        return self.config.safety
    
    def get_vision_config(self) -> VisionConfig:
        """Get vision configuration."""
        return self.config.vision
    
    def get_grounding_config(self) -> GroundingConfig:
        """Get grounding configuration."""
        return self.config.grounding
    
    def get_speech_config(self) -> SpeechConfig:
        """Get speech configuration."""
        return self.config.speech
    
    def get_retrieval_config(self) -> RetrievalConfig:
        """Get retrieval configuration."""
        return self.config.retrieval
    
    def get_scoring_config(self) -> ScoringConfig:
        """Get scoring configuration."""
        return self.config.scoring
    
    def get_coaching_config(self) -> CoachingConfig:
        """Get coaching configuration."""
        return self.config.coaching


def create_default_config_file(file_path: Path = None) -> None:
    """Create a default configuration file."""
    if file_path is None:
        file_path = Path("tutoring_config.json")
    
    config = TutoringConfig.default()
    config.save(file_path)
    print(f"Default configuration created at {file_path}")


def load_config(file_path: Path = None) -> TutoringConfig:
    """Load configuration from file."""
    if file_path is None:
        file_path = Path("tutoring_config.json")
    
    return TutoringConfig.load(file_path)


if __name__ == "__main__":
    # Create default config file
    create_default_config_file()
    print("Default configuration file created!")
