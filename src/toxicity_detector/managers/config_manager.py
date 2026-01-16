"""ConfigManager for handling pipeline configuration file operations."""

import os
import yaml
from glob import glob
from typing import Dict, List, Optional
from huggingface_hub import HfFileSystem
from loguru import logger
import pandas as pd

from toxicity_detector.config import PipelineConfig


class ConfigManager:
    """Handles all configuration file operations (local and HuggingFace)."""

    def __init__(
        self,
        local_serialization: bool,
        local_base_path: str | None,
        hf_base_path: str | None,
        hf_key_name: str | None,
        config_path: str,
        default_pipeline_config_file: str,
        pipeline_config_version: str | None = None
    ):
        """Initialize ConfigManager.

        Args:
            local_serialization: If True, use local filesystem; if False,
                use HuggingFace
            local_base_path: Base path for local serialization
            hf_base_path: Base path for HuggingFace serialization (e.g., 'user/repo')
            hf_key_name: Environment variable name containing HF API token
            config_path: Relative path to config files from base_path
            default_pipeline_config_file: Default pipeline config filename
        """
        self.local_serialization = local_serialization
        # Normalize paths to avoid issues with ".", ".." etc.
        self.local_base_path = self._normalize_local_base_path(local_base_path)
        self.hf_base_path = self._normalize_hf_base_path(hf_base_path)
        self.hf_key_name = hf_key_name
        self.config_path = self._normalize_config_path(config_path)
        self.default_pipeline_config_file = default_pipeline_config_file
        self.pipeline_config_version = pipeline_config_version

        self._fs: Optional[HfFileSystem] = None

        # Validate configuration parameters
        self._validate_paths()
        logger.debug(
            "ConfigManager initialized with parameters: "
            f"local_serialization={self.local_serialization}, "
            f"local_base_path={self.local_base_path}, "
            f"hf_base_path={self.hf_base_path}, "
            f"hf_key_name={self.hf_key_name}, "
            f"config_path={self.config_path}, "
            f"default_pipeline_config_file={self.default_pipeline_config_file}, "
            f"pipeline_config_version={self.pipeline_config_version}"
        )

    def _normalize_local_base_path(self, path: str | None) -> str | None:
        """Normalize local base path to resolve '.', '..' etc.

        Args:
            path: Raw local base path that may contain relative components

        Returns:
            Normalized path or None if input was None
        """
        if path is None:
            return None

        # Normalize the path (resolves '.', '..')
        normalized = os.path.normpath(path)

        # Convert to forward slashes for consistency
        normalized = normalized.replace("\\", "/")

        # Warn if path contains '..' as it may be unexpected
        if ".." in path:
            logger.warning(
                f"local_base_path '{path}' contains '..' which "
                f"resolves to: '{normalized}'. "
                f"Ensure this is the intended directory."
            )

        logger.debug(f"Normalized local_base_path from '{path}' to '{normalized}'")
        return normalized

    def _normalize_hf_base_path(self, path: str | None) -> str | None:
        """Normalize HuggingFace base path to resolve '.', '..' etc.

        Args:
            path: Raw HF base path that may contain relative components

        Returns:
            Normalized path or None if input was None
        """
        if path is None:
            return None

        # First normalize the path structure
        normalized = os.path.normpath(path)

        # Convert to forward slashes (required for HF)
        normalized = normalized.replace("\\", "/")

        # Remove leading './' if present
        if normalized.startswith("./"):
            normalized = normalized[2:]

        # Warn if path contains '..' - this is problematic for HF paths
        if ".." in path:
            logger.warning(
                f"hf_base_path '{path}' contains '..' which is resolved "
                f"to: '{normalized}'. "
                f"This may not work as expected with HuggingFace repositories. "
                f"Consider using absolute repository paths like 'user/repo'."
            )

        logger.debug(f"Normalized hf_base_path from '{path}' to '{normalized}'")
        return normalized

    def _normalize_config_path(self, config_path: str) -> str:
        """Normalize config path to remove '.', '..' and ensure clean path.

        Args:
            config_path: Raw config path that may contain '.' or '..'

        Returns:
            Normalized path safe for both local and HF use
        """
        # Normalize the path (resolves '.', '..')
        normalized = os.path.normpath(config_path)

        # Convert to forward slashes for consistency (works on all platforms)
        normalized = normalized.replace("\\", "/")

        # Remove leading './' if present
        if normalized.startswith("./"):
            normalized = normalized[2:]

        # Warn if path tries to escape with '..'
        if normalized.startswith("..") or "/.." in normalized:
            logger.warning(
                f"config_path '{config_path}' contains '..' which may cause "
                f"unexpected behavior, especially with HuggingFace paths. "
                f"Normalized to: '{normalized}'"
            )

        # Special case: if just ".", treat as empty string (current directory)
        if normalized == ".":
            normalized = ""

        logger.debug(f"Normalized config_path from '{config_path}' to '{normalized}'")
        return normalized

    def _validate_paths(self):
        """Validate configuration paths based on serialization type."""
        if self.local_serialization:
            if not self.local_base_path:
                raise ValueError(
                    "local_base_path parameter must be set when "
                    "local_serialization is True."
                )
            if self.hf_base_path:
                logger.warning(
                    "hf_base_path parameter will be ignored since "
                    "local_serialization is True."
                )
        else:
            if not self.hf_base_path:
                raise ValueError(
                    "hf_base_path parameter must be set when "
                    "local_serialization is False."
                )
            if not self.hf_key_name or len(self.hf_key_name.strip()) == 0:
                logger.warning(
                    "hf_key_name not set for accessing HuggingFace path "
                    "Authentication may fail if the repository is private."
                )
            if self.local_base_path:
                logger.warning(
                    "local_base_path parameter will be ignored since "
                    "local_serialization is False."
                )

    @property
    def fs(self) -> HfFileSystem:
        """Lazy-load HuggingFace filesystem."""
        if not self.local_serialization and self._fs is None:
            token = os.environ.get(self.hf_key_name) if self.hf_key_name else None
            self._fs = HfFileSystem(token=token)
        return self._fs

    def _normalize_hf_path(self, path: str) -> str:
        """Remove 'datasets/' prefix if present to avoid duplication."""
        return path[len("datasets/"):] if path.startswith("datasets/") else path

    def _get_config_base_path(self) -> str:
        """Get the base path where config files are located."""
        if self.local_serialization:
            return os.path.join(
                self.local_base_path,  # type: ignore (checked by _validate_paths)
                self.config_path,  # type: ignore (checked by _validate_paths)
            )
        else:
            base = self._normalize_hf_path(self.hf_base_path)  # type: ignore
            # (checked by _validate_paths)
            return os.path.join(base, self.config_path).replace("\\", "/")

    def _get_full_path(self, file_name: str, for_hf_api: bool = False) -> str:
        """Get full path for a config file.

        Args:
            file_name: Name of the config file
            for_hf_api: If True, prepend 'hf://datasets/' for HF API calls

        Returns:
            Full path to the file
        """
        if self.local_serialization:
            return os.path.join(self._get_config_base_path(), file_name)
        else:
            path = os.path.join(self._get_config_base_path(), file_name).replace(
                "\\", "/"
            )
            return f"hf://datasets/{path}" if for_hf_api else path

    def load_yaml(self, file_name: str) -> Dict:
        """Load YAML file from local or HF storage.

        Args:
            file_name: Name of the YAML file to load

        Returns:
            Dictionary with loaded YAML content
        """
        logger.info(f"Loading YAML file: {file_name}")

        if self.local_serialization:
            path = self._get_full_path(file_name)
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            path = self._get_full_path(file_name, for_hf_api=True)
            with self.fs.open(path, "rb") as f:
                return yaml.safe_load(f)

    def save_yaml(self, file_name: str, data: Dict, make_dirs: bool = False):
        """Save YAML file to local or HF storage.

        Args:
            file_name: Name of the YAML file to save
            data: Dictionary to save as YAML
            make_dirs: If True, create directories if they don't exist
        """
        logger.info(f"Saving YAML file: {file_name}")

        if self.local_serialization:
            path = self._get_full_path(file_name)
            if make_dirs:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        else:
            if make_dirs:
                dir_path = self._get_config_base_path()
                self.fs.makedirs(f"hf://datasets/{dir_path}", exist_ok=True)
            path = self._get_full_path(file_name, for_hf_api=True)
            with self.fs.open(path, "wb") as f:
                yaml_str = yaml.dump(data, allow_unicode=True, default_flow_style=False)
                f.write(yaml_str.encode("utf-8"))

    def load_string(self, file_name: str) -> str:
        """Load text file as string.

        Args:
            file_name: Name of the file to load

        Returns:
            File contents as string
        """
        logger.info(f"Loading string file: {file_name}")

        if self.local_serialization:
            path = self._get_full_path(file_name)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            path = self._get_full_path(file_name, for_hf_api=True)
            with self.fs.open(path, "rb") as f:
                return f.read().decode("utf-8")

    def get_pipeline_config_as_string(self, file_name: str | None = None) -> str:
        """Get pipeline config as string.

        Args:
            file_name: Config filename. If None, loads default config.

        Returns:
            Config file contents as string
        """
        file_name = file_name or self.default_pipeline_config_file
        logger.info(f"Loading pipeline config as string: {file_name}")
        return self.load_string(file_name)

    def save_string(self, file_name: str, content: str):
        """Save string to text file.

        Args:
            file_name: Name of the file to save
            content: String content to save
        """
        logger.info(f"Saving string file: {file_name}")

        if self.local_serialization:
            path = self._get_full_path(file_name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            path = self._get_full_path(file_name, for_hf_api=True)
            with self.fs.open(path, "wb") as f:
                f.write(content.encode("utf-8"))

    def file_exists(self, file_name: str) -> bool:
        """Check if file exists.

        Args:
            file_name: Name of the file to check

        Returns:
            True if file exists, False otherwise
        """
        if self.local_serialization:
            path = self._get_full_path(file_name)
            return os.path.isfile(path)
        else:
            path = self._get_full_path(file_name, for_hf_api=True)
            return self.fs.exists(path)

    def list_config_files(self) -> List[str]:
        """List all pipeline config files, optionally filtered by version.

        Args:
            config_version: If provided, only return configs matching this version

        Returns:
            List of config filenames (without path)
        """
        logger.info(
            f"Listing config files (version filter: {self.pipeline_config_version})"
        )
        config_files = []

        if self.local_serialization:
            base_path = self._get_config_base_path()
            pattern = os.path.join(base_path, "*.yaml")

            for path in glob(pattern):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        # If no version filter, or version matches, add to list
                        if (
                            self.pipeline_config_version is None
                            or (
                                data.get("config_version") ==
                                self.pipeline_config_version
                            )
                        ):
                            config_files.append(os.path.basename(path))
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")
        else:
            base_path = self._get_config_base_path()
            pattern = f"hf://datasets/{base_path}/*.yaml"

            try:
                for path in self.fs.glob(pattern):
                    try:
                        with self.fs.open(path, "rb") as f:
                            data = yaml.safe_load(f)
                            # If no version filter, or version matches, add to list
                            if (
                                self.pipeline_config_version is None
                                or (
                                    data.get("config_version") ==
                                    self.pipeline_config_version
                                )
                            ):
                                config_files.append(os.path.basename(path))
                    except Exception as e:
                        logger.warning(f"Failed to read {path}: {e}")
            except Exception as e:
                logger.error(f"Failed to list files with pattern {pattern}: {e}")

        logger.info(f"Found {len(config_files)} config files")
        return sorted(config_files)

    def load_pipeline_config(self, file_name: str | None = None) -> PipelineConfig:
        """Load pipeline config from file.

        Args:
            file_name: Config filename. If None, loads default config.

        Returns:
            Loaded PipelineConfig object
        """
        file_name = file_name or self.default_pipeline_config_file
        logger.info(f"Loading pipeline config: {file_name}")

        config_dict = self.load_yaml(file_name)
        config = PipelineConfig(**config_dict)
        if (
            self.pipeline_config_version and
            config.config_version != self.pipeline_config_version
        ):
            logger.warning(
                f"Loaded pipeline config '{file_name}' has config_version "
                f"'{config.config_version}' which does not match the expected "
                f"version '{self.pipeline_config_version}'."
            )
        return config

    def get_default_pipeline_config(self) -> PipelineConfig:
        """Load the default pipeline config.

        Returns:
            Default PipelineConfig object
        """
        return self.load_pipeline_config()

    def save_pipeline_config(self, pipeline_config: PipelineConfig, file_name: str):
        """Save pipeline config to file.

        Args:
            pipeline_config: PipelineConfig object to save
            file_name: Filename to save to
        """
        logger.info(f"Saving pipeline config: {file_name}")
        self.save_yaml(file_name, pipeline_config.model_dump())

    def get_config_path(self) -> str:
        """Get the full config path (for backward compatibility).

        Returns:
            Full path to config directory
        """
        return self._get_config_base_path()

    def load_toxicity_example_data(self, data_file_path: str) -> pd.DataFrame:
        """Load toxicity example data from CSV file.

        Args:
            data_file_path: Relative path to CSV file from base_path

        Returns:
            DataFrame with 'text' and 'source' columns
        """
        logger.info(f"Loading toxicity example data from {data_file_path}")

        if self.local_serialization:
            full_path = os.path.join(
                self.local_base_path,  # type: ignore (checked by _validate_paths)
                data_file_path,
            )
            with open(full_path, "r", encoding="utf-8") as f:
                example_data_df = pd.read_csv(f)
        else:
            base = self._normalize_hf_path(self.hf_base_path)  # type: ignore
            # (checked by _validate_paths)
            hf_path = f"hf://datasets/{base}/{data_file_path}".replace("\\", "/")

            token = os.environ.get(self.hf_key_name) if self.hf_key_name else None
            fs = HfFileSystem(token=token)

            with fs.open(hf_path, "rb") as f:
                example_data_df = pd.read_csv(f)

        logger.info("Loading toxicity example data done.")
        return pd.DataFrame(example_data_df[["text", "source"]])
