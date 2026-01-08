from pydantic import BaseModel, Field, field_validator, model_validator
import enum
from typing import Dict, Self, Optional, Any
import yaml
from huggingface_hub import HfFileSystem
import os
from loguru import logger
from datetime import datetime
from functools import lru_cache
from importlib.resources import files

from toxicity_detector.datamodels import Toxicity, ToxicityType

# Minimum required pipeline config version
MIN_PIPELINE_CONFIG_VERSION = "v0.3"


@lru_cache(maxsize=1)
def _load_default_config_dict() -> Dict[str, Any]:
    """Load default configuration from YAML. Cached for performance."""
    try:
        # Use importlib.resources to access package data
        config_file = files("toxicity_detector.package_data").joinpath(
            "default_pipeline_config.yaml"
        )

        with config_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, AttributeError) as e:
        logger.error(
            f"Failed to load default configuration from package data: {e}. "
            "Ensure package_data is properly installed."
        )
        raise


def _get_default_for_field(field_name: str) -> Any:
    """Get default value for a specific field from YAML."""
    config_dict = _load_default_config_dict()
    return config_dict.get(field_name)


def _get_default_toxicities() -> Dict[str, Toxicity]:
    """
    Get default toxicities from YAML and convert to Toxicity instances.
    
    This is needed because when using default_factory with a lambda that returns
    a dict, Pydantic doesn't perform validation/conversion to model instances.
    We explicitly convert the raw YAML dict to Toxicity instances here.
    """
    toxicities_dict = _get_default_for_field("toxicities")
    if not toxicities_dict:
        return {}
    
    # Convert each toxicity dict to a Toxicity model instance
    return {
        key: Toxicity(**value) for key, value in toxicities_dict.items()
    }


class SubdirConstruction(enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

    @classmethod
    def value2formatcode(cls) -> Dict[str, str]:
        return {
            cls.DAILY.value: "%Y_%m_%d",
            cls.WEEKLY.value: "y%Y_w%W",
            cls.MONTHLY.value: "y%Y_m%m",
            cls.YEARLY.value: "y%Y",
        }


class PipelineConfig(BaseModel):
    local_serialization: bool = Field(
        default_factory=lambda: _get_default_for_field("local_serialization") or True
    )
    hf_base_path: str | None = Field(
        default_factory=lambda: _get_default_for_field("hf_base_path")
    )
    hf_key_name: str | None = Field(
        default_factory=lambda: _get_default_for_field("hf_key_name")
    )
    result_data_path: str = Field(
        default_factory=lambda: (
            _get_default_for_field("result_data_path") or "result_data"
        )
    )
    local_base_path: str | None = Field(
        default_factory=lambda: _get_default_for_field("local_base_path")
    )
    log_path: str = Field(
        default_factory=lambda: _get_default_for_field("log_path") or "logs"
    )
    # one of monthly, weekly, yearly, daily, None
    subdirectory_construction: Optional[str] = Field(
        default_factory=lambda: (_get_default_for_field("subdirectory_construction"))
    )
    toxicities: Dict[str, Toxicity] = Field(default_factory=_get_default_toxicities)
    config_version: str = Field(
        default_factory=lambda: _get_default_for_field("config_version") or "v0.4"
    )
    used_chat_model: str
    description: str | None = Field(
        default_factory=lambda: _get_default_for_field("description")
    )
    system_prompt: str = Field(
        default_factory=lambda: _get_default_for_field("system_prompt")
        or (
            "You are a helpful assistant and an expert for the "
            "categorisation and annotation of texts.\n"
            "You read instructions carefully and follow them precisely.\n"
            "You give concise and clear answers."
        )
    )
    models: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: _get_default_for_field("models") or {}
    )
    env_file: str | None = Field(
        default_factory=lambda: _get_default_for_field("env_file")
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._configure_logging()

    def _configure_logging(self):
        """Configure loguru file handler for logging."""
        if self.local_serialization:
            log_dir_path = os.path.join(
                self.get_base_path(),
                self.log_path,
            )
            os.makedirs(log_dir_path, exist_ok=True)

            # Create log file name with current date
            now = datetime.now()
            log_file_name = now.strftime("toxicity_detector_log_%Y_%m_%d.log")
            log_file_path = os.path.join(log_dir_path, log_file_name)

            # Add file handler to loguru
            # Use rotation="1 day" to create a new file each day
            logger.add(
                log_file_path,
                rotation="1 day",
                retention="30 days",
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
                    "{name}:{function}:{line} - {message}"
                ),
                level="INFO",
            )
            logger.info(f"Configured logging to file: {log_file_path}")

    @model_validator(mode="after")
    def load_env_file(self) -> Self:
        if self.env_file is None:
            logger.warning(
                "No environment file with API keys specified for Pipeline Config."
                " Please set 'env_file' to a valid path if you want "
                "to load environment variables from a file."
            )
        else:
            # check if the env file exists
            from os import path
            env_file_path = path.join(
                self.get_base_path(),
                self.env_file,
            )

            if not path.exists(env_file_path):
                err_msg = (
                    f"Environment file '{env_file_path}' does not exist. "
                    "Please provide a valid path to the environment file. "
                    "Or set it to None if you don't need it and set the "
                    "API keys in other ways as environment variables."
                )
                logger.warning(err_msg)
            else:
                # load the env file
                from dotenv import load_dotenv

                load_dotenv(env_file_path)
                logger.info(f"Loaded environment variables from '{env_file_path}'")
        return self

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        allowed_values = set(self.models.keys())
        if len(allowed_values) == 0:
            raise ValueError("At least one model must be specified under models")
        elif self.used_chat_model not in allowed_values:
            raise ValueError(
                f"used_chat_model must be one of {allowed_values}, "
                f"got {self.used_chat_model}"
            )
        return self

    @field_validator("toxicities")
    def validate_toxicity_types(cls, v):
        allowed_values = ToxicityType._value2member_map_.keys()
        toxicity_types = v.keys()
        if not set(toxicity_types).issubset(set(allowed_values)):
            raise ValueError(
                f"Allowed toxicities are {set(allowed_values)}, "
                f"got {set(toxicity_types)}"
            )
        return v

    @field_validator("config_version")
    @classmethod
    def validate_config_version(cls, v):
        if v is None:
            raise ValueError(
                f"config_version must be specified. "
                f"Minimum required version is {MIN_PIPELINE_CONFIG_VERSION}"
            )

        # Extract version numbers from format "vX.Y" or "vX.Y.Z"
        def parse_version(version_str: str) -> tuple:
            # Remove 'v' prefix and split by '.'
            version_parts = version_str.lstrip("v").split(".")
            return tuple(int(part) for part in version_parts)

        try:
            current_version = parse_version(v)
            min_version = parse_version(MIN_PIPELINE_CONFIG_VERSION)

            if current_version < min_version:
                raise ValueError(
                    f"config_version {v} is below minimum required version "
                    f"{MIN_PIPELINE_CONFIG_VERSION}"
                )
        except (ValueError, AttributeError) as e:
            if "invalid literal" in str(e):
                raise ValueError(
                    f"Invalid config_version format: {v}. "
                    f"Expected format like 'v0.3' or 'v0.4.1'"
                )
            raise

        return v

    @field_validator("subdirectory_construction")
    @classmethod
    def validate_subdirectory_construction(cls, v):
        allowed_values = SubdirConstruction._value2member_map_.keys()
        if (v is not None) and (v not in allowed_values):
            raise ValueError(
                f"subdirectory_construction must be one "
                f"of {set(allowed_values)}, got {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_serialization(self) -> Self:
        if self.local_serialization and self.local_base_path is None:
            raise ValueError(
                "local_base_path must be set if choosing local serialization"
            )
        elif not self.local_serialization and self.hf_base_path is None:
            raise ValueError(
                "hf_base_path must be set if not choosing local serialization"
            )
        return self

    def get_base_path(self) -> str:
        if self.local_serialization:
            assert self.local_base_path is not None
            return self.local_base_path
        else:
            assert self.hf_base_path is not None
            return self.hf_base_path

    @staticmethod
    def from_file(file_path: str) -> "PipelineConfig":
        with open(file_path, encoding="utf-8") as f:
            config = PipelineConfig(**yaml.safe_load(f.read()))
        return config

    @staticmethod
    def from_hf(file_path: str) -> "PipelineConfig":
        fs = HfFileSystem()
        with fs._open(file_path, "rb", encoding="utf-8") as f:
            return PipelineConfig(**yaml.safe_load(f.read()))


# TODO: remove defaults
class UITexts(BaseModel):
    trigger_warning: dict[str, str] = {
        "message": (
            "# Benutzunghinweise (*!vorläufige Formulierung!*)\n\n"
            "+ **Triggerwarnung:** Die Toxizitätdetektorapp enthält in Form "
            "von Beispielen Inhalte, die anstößig oder beunruhigend sein können. "
            "Alle Materialien dienen der Unterstützung von Forschungsarbeiten zur "
            "Verbesserung der Methoden zur Erkennung von Toxizität. Die enthaltenen "
            "Beispiele für Toxizität geben insbesondere nicht wieder, wie die Autoren "
            "über bestimmte Identitätsgruppen bzw. Personen denken. Die Beispiele "
            "stammen aus dem Korpus ...\n"
            "+ **Datenerhebung:** Die eingegebenen Textbeispiele und generierten "
            "Kategorisierungen werden für Forschungszwecke gespeichert und benutzt "
            "um die Performance von Modelle zu steigern. Darüberhinaus werden keine "
            "Daten gesammelt, insbesondere keine personenbezogenen Daten (sofern keine "
            "personenbezogenen Daten in das Textfeld eingetragen werden)\n"
        ),
    }
    app_head: str = (
        "# 📣 Detektor für toxische Sprache\n"
        "In dieser Demoapp kannst Du ausprobieren, wie gut Large Language Models "
        "Toxizität detektieren können.\n"
    )


class AppConfig(BaseModel):
    developer_mode: bool = False
    pipeline_config_version: str = "v0.3"
    default_pipeline_config_file: str = "model_config_cv0.3_2024-11-24_v0.3.yaml"
    local_pipeline_config: bool = True
    local_pipeline_config_base_path: str | None = None
    hf_pipeline_config_base_path: str | None = None
    toxicity_examples_hf_base_path: str = "toxicity_example_data"
    toxicity_examples_data_file: str = "toxicity_examples_detox_germeval21_n10000.csv"
    toxicity_examples_key_name: str | None = None
    pipeline_config_key_name: str | None = None
    feedback: Dict[str, Any] = {
        "likert_scale": {
            "absolutely_correct": "Stimmt absolut",
            "correct": "Stimmt",
            "suspension": "Unklar",
            "incorrect": "Stimmt nicht",
            "absolutely_incorrect": "Stimmt überhaupt nicht",
        }
    }
    ui_texts: UITexts = UITexts()
    env_file: str | None = None

    @model_validator(mode="after")
    def load_env_file(self) -> Self:
        if self.env_file is None:
            logger.warning(
                "No environment file with API keys specified for App Config."
                " Please set 'env_file' to a valid path if you want "
                "to load environment variables from a file."
            )
        else:
            # check if the env file exists
            from os import path

            if not path.exists(self.env_file):
                err_msg = (
                    f"Environment file '{self.env_file}' does not exist. "
                    "Please provide a valid path to the environment file. "
                    "Or set it to None if you don't need it and set the "
                    "API keys in other ways as environment variables."
                )
                logger.warning(err_msg)
            else:
                # load the env file
                from dotenv import load_dotenv

                load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from '{self.env_file}'")
        return self

    @model_validator(mode="after")
    def validate_toxicity_examples_key_name(self) -> Self:
        if (
            not self.toxicity_examples_key_name
            or len(self.toxicity_examples_key_name.strip()) == 0
        ):
            logger.warning(
                f"""
                toxicity_examples_key_name not set for accessing \
                toxicity_examples_hf_base_path '{self.toxicity_examples_hf_base_path}'
                """
            )
        return self

    def get_pipeline_config_path(self) -> str:
        if self.local_pipeline_config:
            base_path = self.local_pipeline_config_base_path
        else:
            base_path = self.hf_pipeline_config_base_path
        assert base_path is not None
        return base_path

    def get_default_pipeline_config(self) -> PipelineConfig:
        if self.local_pipeline_config:
            file_path = os.path.join(
                self.get_pipeline_config_path(), self.default_pipeline_config_file
            )
            return PipelineConfig.from_file(file_path)
        else:
            if self.pipeline_config_key_name:
                fs = HfFileSystem(token=os.environ[self.pipeline_config_key_name])
            else:
                fs = HfFileSystem()
            file_path = os.path.join(
                "hf://datasets",
                self.get_pipeline_config_path(),
                self.default_pipeline_config_file,
            ).replace("\\", "/")
            with fs._open(file_path, "rt", encoding="utf_8") as file:
                return PipelineConfig(**yaml.safe_load(file))

    @staticmethod
    def from_file(file_path: str) -> "AppConfig":
        with open(file_path, encoding="utf_8") as f:
            config = AppConfig(**yaml.safe_load(f))
        return config

    @model_validator(mode="after")
    def validate_pipeline_config_base_path(self) -> Self:
        if self.local_pipeline_config:
            if not self.local_pipeline_config_base_path:
                raise ValueError(
                    "local_pipeline_config_base_path parameter must be set."
                )
            if self.hf_pipeline_config_base_path:
                logger.warning(
                    "hf_pipeline_config_base_path parameter will be ignored."
                )
        else:
            if not self.hf_pipeline_config_base_path:
                raise ValueError("hf_pipeline_config_base_path must be set.")
            if (
                not self.pipeline_config_key_name
                or len(self.pipeline_config_key_name.strip()) == 0
            ):
                logger.warning(
                    f"""
                    pipeline_config_key_name not set for accessing \
                    hf_pipeline_config_base_path '{self.hf_pipeline_config_base_path}'
                    """
                )
            if self.local_pipeline_config_base_path:
                logger.warning(
                    "local_pipeline_config_base_path parameter will be ignored."
                )
        return self
