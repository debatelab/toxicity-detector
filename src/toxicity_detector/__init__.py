"""Toxicity Detector - An LLM-based pipeline to detect toxic speech."""

# Import from chains module
from .chains import (
    BaseChainBuilder,
    IdentifyToxicContentZeroShotChain,
    MonoModelDetectToxicityChain,
    IdentifyToxicContentChatChain,
)

# Import from backend module
from .backend import (
    detect_toxicity,
    ZeroShotClassifier,
    update_feedback,
    save_result,
    get_openai_chat_model,
)

from .result import (
    ToxicityDetectorResult,
)

from .config import (
    AppConfig,
    PipelineConfig,
    SubdirConstruction,
)

from .managers import (
    ConfigManager,
)

from .datamodels import (
    ToxicityType,
    ToxicityAnswer,
    Toxicity,
    Task
)

__all__ = [
    # Chain classes
    "BaseChainBuilder",
    "IdentifyToxicContentZeroShotChain",
    "MonoModelDetectToxicityChain",
    "IdentifyToxicContentChatChain",
    # Backend classes
    "ZeroShotClassifier",
    # Backend functions
    "detect_toxicity",
    "update_feedback",
    "save_result",
    "get_openai_chat_model",
    # Config, output and other basic classes
    "ToxicityType",
    "ToxicityAnswer",
    "Toxicity",
    "Task",
    "ToxicityDetectorResult",
    "PipelineConfig",
    "AppConfig",
    "SubdirConstruction",
    "ConfigManager",
]
