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
    get_toxicity_example_data,
    dump_pipeline_config_str,
    config_file_exists,
    pipeline_config_as_string,
    pipeline_config_file_names,
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

from .datamodels import (
    ToxicityType,
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
    "get_toxicity_example_data",
    "dump_pipeline_config_str",
    "config_file_exists",
    "pipeline_config_as_string",
    "pipeline_config_file_names",
    "update_feedback",
    "save_result",
    "get_openai_chat_model",
    # Config, output and other basic classes
    "ToxicityType",
    "Toxicity",
    "Task",
    "ToxicityDetectorResult",
    "PipelineConfig",
    "AppConfig",
    "SubdirConstruction"
]
