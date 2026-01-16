"""Configuration loader for the Gradio app."""

import os
import gradio as gr
from pathlib import Path

from toxicity_detector import ConfigManager
from toxicity_detector.config import AppConfig, PipelineConfig

# Global variables to hold app config (will be initialized by init_app_config)
app_config: AppConfig | None = None
config_manager: ConfigManager | None = None


def _config_manager() -> ConfigManager:
    """Get the initialized ConfigManager instance."""
    if config_manager is None:
        raise RuntimeError(
            "ConfigManager not initialized. Call init_app_config() first."
        )
    return config_manager


def _app_config() -> AppConfig:
    """Get the initialized AppConfig instance."""
    if app_config is None:
        raise RuntimeError("AppConfig not initialized. Call init_app_config() first.")
    return app_config


def init_app_config(config_path=None, config_type="app"):
    """Initialize app configuration and config manager.

    Args:
        config_path: Path to configuration file. If None, uses default path.
        config_type: Either "app" for AppConfig or "pipeline" for PipelineConfig.
                     If "pipeline", creates AppConfig with defaults.
    """
    global app_config, config_manager

    # Load app config file path from environment variable if not provided
    if config_path is None:
        config_path = os.getenv("TOXICITY_DETECTOR_APP_CONFIG_FILE")
        if config_path is None:
            raise ValueError(
                "No config_path provided and "
                "TOXICITY_DETECTOR_APP_CONFIG_FILE not set as env!"
            )
        if config_type == "pipeline":
            raise ValueError(
                "config_type 'pipeline' cannot be used "
                "when config_path is from env variable!"
            )

    if config_type == "app":
        app_config = AppConfig.from_file(config_path)
        # re-init None values from pipeline config values and validate
        pipeline_config = PipelineConfig.from_file(
            os.path.join(
                app_config.config_path,
                app_config.pipeline_config_file
            )
        )
        if app_config.local_serialization is None:
            if pipeline_config.local_serialization is None:
                raise gr.Error(
                    "local_serialization must be set in either "
                    "app config or pipeline config!"
                )
            app_config.local_serialization = pipeline_config.local_serialization
        if app_config.local_base_path is None:
            if (
                app_config.local_serialization
                and pipeline_config.local_base_path is None
            ):
                raise gr.Error(
                    "local_base_path must be set in either "
                    "app config or pipeline config when "
                    "local_serialization is True!"
                )
            app_config.local_base_path = pipeline_config.local_base_path
        if app_config.hf_base_path is None:
            if (
                not app_config.local_serialization
                and pipeline_config.hf_base_path is None
            ):
                raise gr.Error(
                    "hf_base_path must be set in either "
                    "app config or pipeline config when "
                    "local_serialization is False!"
                )
            app_config.hf_base_path = pipeline_config.hf_base_path
        if app_config.hf_key_name is None:
            app_config.hf_key_name = pipeline_config.hf_key_name
        if app_config.env_file is not None:
            app_config.env_file = pipeline_config.env_file
            app_config.load_env_file()

        config_manager = ConfigManager(
            app_config.local_serialization,
            app_config.local_base_path,
            app_config.hf_base_path,
            app_config.hf_key_name,
            app_config.config_path,
            app_config.pipeline_config_file,
            app_config.pipeline_config_version,
        )
    elif config_type == "pipeline":
        # Load pipeline config, create AppConfig with defaults
        # and initialize ConfigManager with values from pipeline config
        pipeline_config = PipelineConfig.from_file(config_path)
        # Create minimal AppConfig with defaults
        # (We do not need pass serialization configuration since we
        # pass it directly to the ConfigManager.)
        app_config = AppConfig(pipeline_config_file=Path(config_path).name)
        # Initialize ConfigManager
        config_manager = ConfigManager(
            pipeline_config.local_serialization,
            pipeline_config.local_base_path,
            pipeline_config.hf_base_path,
            pipeline_config.hf_key_name,
            app_config.config_path,  # defaults to "configs"
            Path(config_path).name,  # setting the passed pipeline config as default
            pipeline_config.config_version,  # version from loaded pipeline config
        )
    else:
        raise ValueError(
            f"Invalid config_type: {config_type}. Must be 'app' or 'pipeline'."
        )
