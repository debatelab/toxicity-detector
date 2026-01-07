import pytest
from pydantic import ValidationError
from toxicity_detector.config import PipelineConfig, MIN_PIPELINE_CONFIG_VERSION


def get_minimal_config_data(config_version: str) -> dict:
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
        
        assert "config_version v0.2 is below minimum required version" in str(exc_info.value)
    
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
        
        assert "config_version must be specified" in str(exc_info.value)
    
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
