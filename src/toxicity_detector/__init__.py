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
    log_message,
    get_toxicity_example_data,
    dump_model_config_str,
    config_file_exists,
    model_config_as_string,
    model_config,
    model_config_file_names,
    update_feedback,
    save_result,
    get_request_uuid,
    get_openai_chat_model,
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
    "log_message",
    "get_toxicity_example_data",
    "dump_model_config_str",
    "config_file_exists",
    "model_config_as_string",
    "model_config",
    "model_config_file_names",
    "update_feedback",
    "save_result",
    "get_request_uuid",
    "get_openai_chat_model",
]
