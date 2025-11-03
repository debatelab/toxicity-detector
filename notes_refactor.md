# Refactoring the Toxicity Pipeline

## Tasks

- [ ] Refactor model configuration handling to improve clarity and maintainability.
    - [ ] Rename model-config to pipeline-config.
    - [ ] Wrap config parameters into dedicated Pydantic classes.
    - [ ] Add Pydantic validation for config parameters.
    - [ ] Refactor "toxicity_type" as an Enum.
    - [ ] Better separation of pipeline-specific and app-specific configurations. (see tasks in `app_config.yaml`)
    - [ ] Provide default config as pkg-data (only for pipeline-config, not app-config).
- [ ] Refactor gradio app
    - [ ] Move serialization of result data serialisation into the backend (?)
    - [ ] Modularize components for better reusability.
- [ ] Logging and data serialization
    - [ ] Use loguru as logging library throughout the codebase.
        - [ ] Perhaps a dedicated handler for logging on HF datasets?
    - [ ] Refactor data serialization split (local vs. serialization on HF hub). (The code is not aweful).