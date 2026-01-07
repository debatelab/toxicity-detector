import pytest
from pydantic import ValidationError
from toxicity_detector.config import (
    PipelineConfig,
    MIN_PIPELINE_CONFIG_VERSION,
    _load_default_config_dict,
    _get_default_for_field,
    _get_default_toxicities,
)
from toxicity_detector.datamodels import Toxicity, Task


def get_minimal_config_data(config_version: str | None) -> dict:
    """Helper function to get minimal valid config data"""
    return {
        "config_version": config_version,
        "used_chat_model": "test_model",
        "models": {"test_model": {"name": "Test Model"}},
        "local_serialization": False,  # Avoid needing local_base_path
        "hf_base_path": "test/path",
        "env_file": None,  # Avoid env file warnings
    }


class TestPipelineConfigVersionValidator:
    """Test suite for PipelineConfig version validation"""

    def test_valid_version_equal_to_minimum(self):
        """Config version equal to minimum should be valid"""
        config_data = get_minimal_config_data("v0.3")
        config = PipelineConfig(**config_data)
        assert config.config_version == "v0.3"

    def test_valid_version_higher_than_minimum(self):
        """Config version higher than minimum should be valid"""
        config_data = get_minimal_config_data("v0.4")
        config = PipelineConfig(**config_data)
        assert config.config_version == "v0.4"

    def test_valid_version_with_patch(self):
        """Config version with patch number should be valid"""
        config_data = get_minimal_config_data("v0.4.1")
        config = PipelineConfig(**config_data)
        assert config.config_version == "v0.4.1"

    def test_valid_version_much_higher(self):
        """Config version much higher than minimum should be valid"""
        config_data = get_minimal_config_data("v0.10")
        config = PipelineConfig(**config_data)
        assert config.config_version == "v0.10"

    def test_invalid_version_below_minimum(self):
        """Config version below minimum should raise ValidationError"""
        config_data = get_minimal_config_data("v0.2")
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(**config_data)

        assert "config_version v0.2 is below minimum required version" in str(
            exc_info.value
        )

    def test_invalid_version_much_lower(self):
        """Config version much lower than minimum should raise ValidationError"""
        config_data = get_minimal_config_data("v0.1")
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(**config_data)

        assert "is below minimum required version" in str(exc_info.value)

    def test_missing_version(self):
        """Missing config_version should raise ValidationError"""
        config_data = get_minimal_config_data(None)
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(**config_data)

        # When None is explicitly passed, Pydantic validates it as invalid string
        assert "string_type" in str(exc_info.value)

    def test_invalid_version_format_no_v_prefix(self):
        """Version without 'v' prefix should work since we strip the 'v' prefix"""
        config_data = get_minimal_config_data("0.3")
        config = PipelineConfig(**config_data)
        assert config.config_version == "0.3"

    def test_invalid_version_format_text(self):
        """Non-numeric version should raise ValidationError"""
        config_data = get_minimal_config_data("v0.abc")
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(**config_data)

        assert "Invalid config_version format" in str(exc_info.value)

    def test_version_comparison_with_different_lengths(self):
        """Version v0.3.0 should equal v0.3 when comparing"""
        # v0.3.0 should be considered >= v0.3
        config_data = get_minimal_config_data("v0.3.0")
        config = PipelineConfig(**config_data)
        assert config.config_version == "v0.3.0"

    def test_constant_value(self):
        """MIN_PIPELINE_CONFIG_VERSION should be v0.3"""
        assert MIN_PIPELINE_CONFIG_VERSION == "v0.3"


class TestDefaultConfigLoader:
    """Test suite for default config loading from package data"""

    def test_load_default_config_dict_returns_dict(self):
        """_load_default_config_dict should return a dictionary"""
        config_dict = _load_default_config_dict()
        assert isinstance(config_dict, dict)
        assert len(config_dict) > 0

    def test_load_default_config_dict_caching(self):
        """_load_default_config_dict should return the same cached object"""
        config_dict1 = _load_default_config_dict()
        config_dict2 = _load_default_config_dict()
        # Should be the exact same object due to caching
        assert config_dict1 is config_dict2

    def test_load_default_config_dict_has_expected_keys(self):
        """Default config should have expected keys"""
        config_dict = _load_default_config_dict()
        expected_keys = [
            "config_version",
            "local_serialization",
            "result_data_path",
            "log_path",
            "subdirectory_construction",
            "toxicities",
            "system_prompt",
            "description",
        ]
        for key in expected_keys:
            assert key in config_dict, f"Expected key '{key}' not found in config"

    def test_get_default_for_field_returns_correct_values(self):
        """_get_default_for_field should return correct values from YAML"""
        assert _get_default_for_field("config_version") == "v0.4"
        assert _get_default_for_field("local_serialization") is True
        assert _get_default_for_field("result_data_path") == "result_data"
        assert _get_default_for_field("log_path") == "logs"
        assert _get_default_for_field("subdirectory_construction") == "daily"

    def test_get_default_for_field_returns_none_for_missing_key(self):
        """_get_default_for_field should return None for non-existent keys"""
        assert _get_default_for_field("nonexistent_key") is None

    def test_get_default_for_field_toxicities_structure(self):
        """Toxicities should have proper structure"""
        toxicities = _get_default_for_field("toxicities")
        assert isinstance(toxicities, dict)
        assert len(toxicities) > 0
        # Check expected toxicity types
        assert "personalized_toxicity" in toxicities
        assert "hatespeech" in toxicities

    def test_get_default_toxicities_returns_toxicity_instances(self):
        """_get_default_toxicities should return Dict[str, Toxicity] instances"""
        toxicities = _get_default_toxicities()
        assert isinstance(toxicities, dict)
        assert len(toxicities) > 0
        
        # All values should be Toxicity instances, not dicts
        for key, value in toxicities.items():
            assert isinstance(
                value, Toxicity
            ), f"toxicities['{key}'] should be Toxicity instance, got {type(value)}"
            # Verify Toxicity has required fields
            assert isinstance(value.title, str)
            assert isinstance(value.user_description, str)
            assert isinstance(value.llm_description, str)
            assert isinstance(value.tasks, dict)

    def test_get_default_toxicities_structure_is_correct(self):
        """Toxicities should have correct nested structure with Task instances"""
        toxicities = _get_default_toxicities()
        
        # Check personalized_toxicity structure
        assert "personalized_toxicity" in toxicities
        pers_tox = toxicities["personalized_toxicity"]
        assert pers_tox.title == "Personalisierte Toxizität"
        assert len(pers_tox.tasks) > 0
        
        # Check that tasks contain Task instances
        for task_category, tasks_dict in pers_tox.tasks.items():
            assert isinstance(tasks_dict, dict)
            for task_name, task in tasks_dict.items():
                assert isinstance(
                    task, Task
                ), f"Task '{task_name}' should be Task instance, got {type(task)}"
                assert hasattr(task, "name")
                assert hasattr(task, "llm_description")

    def test_get_default_toxicities_is_not_cached(self):
        """_get_default_toxicities should return new Toxicity instances each call"""
        toxicities1 = _get_default_toxicities()
        toxicities2 = _get_default_toxicities()
        
        # Dicts should be different objects
        assert toxicities1 is not toxicities2
        # But the Toxicity instances inside might be the same or different
        # depending on how Pydantic creates them


class TestPipelineConfigDefaultValues:
    """Test suite for PipelineConfig initialization with defaults"""

    def test_config_uses_yaml_defaults_when_not_specified(self):
        """Config should use YAML defaults when fields are not provided"""
        config = PipelineConfig(
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        # These should come from the YAML defaults
        assert config.config_version == "v0.4"
        assert config.result_data_path == "result_data"
        assert config.log_path == "logs"
        assert config.subdirectory_construction == "daily"
        assert "You are a helpful assistant" in config.system_prompt

    def test_config_explicit_params_override_defaults(self):
        """Explicitly provided parameters should override YAML defaults"""
        config = PipelineConfig(
            config_version="v0.5",
            result_data_path="custom_results",
            log_path="custom_logs",
            subdirectory_construction="monthly",
            system_prompt="Custom system prompt",
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        # All should be the custom values
        assert config.config_version == "v0.5"
        assert config.result_data_path == "custom_results"
        assert config.log_path == "custom_logs"
        assert config.subdirectory_construction == "monthly"
        assert config.system_prompt == "Custom system prompt"

    def test_config_partial_override_of_defaults(self):
        """Some fields overridden, others use defaults"""
        config = PipelineConfig(
            result_data_path="my_custom_results",
            # log_path not specified - should use default
            config_version="v0.4",
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        assert config.result_data_path == "my_custom_results"  # overridden
        assert config.log_path == "logs"  # from default
        assert config.subdirectory_construction == "daily"  # from default

    def test_config_with_local_base_path_override(self):
        """Test overriding local_base_path (which defaults to '.' in YAML)"""
        import tempfile
        import os
        
        # Use a valid temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                local_base_path=tmpdir,
                local_serialization=True,
                used_chat_model="test_model",
                models={"test_model": {"name": "Test"}},
                env_file=None,
            )
            assert config.local_base_path == tmpdir
            assert config.local_serialization is True

    def test_config_toxicities_uses_yaml_defaults(self):
        """Toxicities should be loaded from YAML by default"""
        config = PipelineConfig(
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        # Should have toxicities from YAML
        assert len(config.toxicities) > 0
        assert "personalized_toxicity" in config.toxicities
        assert "hatespeech" in config.toxicities

    def test_config_toxicities_are_toxicity_instances(self):
        """Toxicities in config should be Toxicity instances, not dicts"""
        config = PipelineConfig(
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        
        # Verify all toxicities are proper Toxicity instances
        for key, value in config.toxicities.items():
            assert isinstance(
                value, Toxicity
            ), f"config.toxicities['{key}'] should be Toxicity, got {type(value)}"
            assert hasattr(value, "title")
            assert hasattr(value, "user_description")
            assert hasattr(value, "llm_description")
            assert hasattr(value, "tasks")
            
            # Verify we can access attributes (not dict keys)
            assert isinstance(value.title, str)
            assert isinstance(value.user_description, str)
            assert isinstance(value.llm_description, str)
            assert isinstance(value.tasks, dict)

    def test_config_toxicities_nested_tasks_are_task_instances(self):
        """Nested tasks in toxicities should be Task instances"""
        config = PipelineConfig(
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        
        # Check a specific toxicity's tasks
        if "personalized_toxicity" in config.toxicities:
            pers_tox = config.toxicities["personalized_toxicity"]
            assert isinstance(pers_tox, Toxicity)
            
            # Check tasks structure
            for task_category, tasks_dict in pers_tox.tasks.items():
                assert isinstance(tasks_dict, dict), (
                    f"Task category '{task_category}' should be dict"
                )
                for task_name, task in tasks_dict.items():
                    assert isinstance(task, Task), (
                        f"Task '{task_name}' should be Task instance, got {type(task)}"
                    )
                    # Verify Task attributes
                    assert isinstance(task.name, str)
                    assert isinstance(task.llm_description, str)

    def test_config_toxicities_can_be_overridden(self):
        """Toxicities can be explicitly overridden with empty dict"""
        # Override with empty dict to not use defaults
        config = PipelineConfig(
            toxicities={},
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        # Should have no toxicities (overridden with empty dict)
        assert len(config.toxicities) == 0
        assert "personalized_toxicity" not in config.toxicities
        assert "hatespeech" not in config.toxicities

    def test_config_toxicities_can_be_provided_as_dicts(self):
        """Toxicities can be provided as dicts and Pydantic will convert them"""
        # Provide a minimal toxicity as a dict, using a valid toxicity type
        # Note: Only toxicity types in ToxicityType enum are allowed
        custom_toxicities_dict = {
            "personalized_toxicity": {
                "title": "Custom Personalized Toxicity",
                "user_description": "User facing description",
                "llm_description": "LLM description",
                "tasks": {
                    "category1": {
                        "task1": {
                            "name": "Task 1",
                            "llm_description": "Task 1 description",
                        }
                    }
                },
            }
        }
        
        config = PipelineConfig(
            toxicities=custom_toxicities_dict,
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        
        # Pydantic should have converted the dict to Toxicity instance
        assert len(config.toxicities) == 1
        assert "personalized_toxicity" in config.toxicities
        
        custom_tox = config.toxicities["personalized_toxicity"]
        assert isinstance(custom_tox, Toxicity)
        assert custom_tox.title == "Custom Personalized Toxicity"
        assert custom_tox.user_description == "User facing description"
        assert custom_tox.llm_description == "LLM description"
        
        # Check nested Task instances
        assert "category1" in custom_tox.tasks
        assert "task1" in custom_tox.tasks["category1"]
        task1 = custom_tox.tasks["category1"]["task1"]
        assert isinstance(task1, Task)
        assert task1.name == "Task 1"
        assert task1.llm_description == "Task 1 description"

    def test_config_toxicities_rejects_invalid_toxicity_types(self):
        """Config should reject toxicity types not in ToxicityType enum"""
        from toxicity_detector.datamodels import ToxicityType
        
        invalid_toxicities_dict = {
            "invalid_toxicity_type": {
                "title": "Invalid Type",
                "user_description": "Description",
                "llm_description": "Description",
                "tasks": {},
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(
                toxicities=invalid_toxicities_dict,
                used_chat_model="test_model",
                models={"test_model": {"name": "Test"}},
                local_serialization=False,
                hf_base_path="test/path",
                env_file=None,
            )
        
        # Should mention allowed values
        error_msg = str(exc_info.value)
        assert "Allowed toxicities" in error_msg or "invalid_toxicity_type" in error_msg

    def test_config_description_defaults_to_yaml(self):
        """Description should use YAML default if not provided"""
        config = PipelineConfig(
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        # Should have description from YAML
        assert config.description is not None
        assert "Konfiguration" in config.description

    def test_config_description_can_be_overridden(self):
        """Description can be explicitly set"""
        config = PipelineConfig(
            description="My custom description",
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        assert config.description == "My custom description"

    def test_config_description_can_be_set_to_none(self):
        """Description can be explicitly set to None"""
        config = PipelineConfig(
            description=None,
            used_chat_model="test_model",
            models={"test_model": {"name": "Test"}},
            local_serialization=False,
            hf_base_path="test/path",
            env_file=None,
        )
        assert config.description is None

    def test_config_empty_models_dict_override(self):
        """Empty models dict should fail validation even if overriding default"""
        with pytest.raises(ValidationError) as exc_info:
            PipelineConfig(
                models={},  # Empty, should fail
                used_chat_model="test_model",
                local_serialization=False,
                hf_base_path="test/path",
                env_file=None,
            )
        assert "At least one model must be specified" in str(exc_info.value)

    def test_config_all_fields_explicit(self):
        """Test creating config with all fields explicitly provided"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                local_serialization=True,
                local_base_path=tmpdir,
                hf_base_path=None,
                hf_key_name=None,
                result_data_path="explicit_results",
                log_path="explicit_logs",
                subdirectory_construction="weekly",
                toxicities={},
                config_version="v0.6",
                used_chat_model="explicit_model",
                description="Explicit description",
                system_prompt="Explicit system prompt",
                models={"explicit_model": {"name": "Explicit Model"}},
                env_file=None,
            )
            assert config.local_serialization is True
            assert config.local_base_path == tmpdir
            assert config.result_data_path == "explicit_results"
            assert config.log_path == "explicit_logs"
            assert config.subdirectory_construction == "weekly"
            assert config.toxicities == {}
            assert config.config_version == "v0.6"
            assert config.used_chat_model == "explicit_model"
            assert config.description == "Explicit description"
            assert config.system_prompt == "Explicit system prompt"
