# 📣 Toxicity Detector

An LLM-based pipeline to detect toxic speech.

## 🎯 About the Toxicity Detector

The Toxicity Detector is a configurable pipeline that uses a Large Language Model (LLM) to analyze a text and decide whether it contains toxic speech.

It supports two toxicity types out of the box:

- **`personalized_toxicity`**: toxic speech directed at a specific individual (insults, threats, harassment, …)
- **`hatespeech`**: group-based toxicity / hate speech (targeting groups or individuals because of group membership)

Both toxicity types are defined in the pipeline configuration file under the `toxicities:` section.

### The Toxicity Detector Workflow

At a high level the pipeline works as follows:

1. **Preprocessing / preparatory analysis**: the model answers “general questions” that help it interpret the input (e.g., who is targeted, irony/quotes/context).
2. **Indicator analysis**: the model evaluates a set of configurable indicators (tasks) that represent typical forms of toxicity (e.g., threats, insults, victim shaming).
3. **Final decision**: the pipeline aggregates these intermediate results and returns:
  - `contains_toxicity`: one of `true`, `false`, `unclear`
  - `analysis_result`: a human-readable explanation

The indicators and the phrasing of the model prompts are configurable via YAML.

<div align="center">
  <p align="center">
  <img src="./img/tode_bg_white.png" alt="Figure of workflow">
  </p>
</div>


## 🖥️ Quick Start

### Prerequisites

- Python 3.12 or higher

### Installation via PyPi

Install the [`toxicity-detector` package](https://pypi.org/project/toxicity-detector/) via PyPi (e.g., by using pip):

```bash
pip install toxicity-detector
```

### Setting up a minimal configuration

You need a **pipeline configuration** (YAML) to run toxicity detection. This repo ships example configs in `config/`:

- [`config/pipeline_config.yaml`](https://github.com/debatelab/toxicity-detector/blob/main/config/pipeline_config.yaml): pipeline configuration used by the CLI and Python API
- [`config/app_config.yaml`](https://github.com/debatelab/toxicity-detector/blob/main/config/app_config.yaml): configuration for the Gradio demo app (optional)

Start by copying the example files and adjusting them to your environment (models, API keys, storage paths).

#### API Keys

API keys are referenced by name in the pipeline config (e.g., `API_KEY_NAME`) and are expected to be present as environment variables.

Create a `.env` file in the project root with the following variables:

```txt
# API Keys (by the names as specified in the model config files)
API_KEY_NAME=your_api_key_value
```

Alternatively, you can set the environment variables in your shell/session (instead of using `.env`).


## 🚀 Running the Pipeline 

### Using the CLI

The simplest way to run toxicity detection from the command line (within the environment you installed the toxicity package into):

```bash
# Basic usage
toxicity-detector detect \
  --text "Your text to analyze" \
  --pipeline-config ./config/pipeline_config.yaml

# With all options
toxicity-detector detect \
  --text "Your text to analyze" \
  --pipeline-config ./config/pipeline_config.yaml \
  --toxicity-type personalized_toxicity \
  --source "chat" \
  --context "Additional context here" \
  --save \
  --verbose
```

### Programmatically

```python
from toxicity_detector import detect_toxicity, PipelineConfig

# Load pipeline configuration from YAML file
pipeline_config = PipelineConfig.from_file('./config/pipeline_config.yaml')

# The text to analyze for toxicity
input_text = 'Peter is dumn.'

# Run toxicity detection
result = detect_toxicity(
    input_text=input_text,  # The text to be analyzed
    user_input_source=None,  # Optional: identifier for the source of the input (e.g., 'chat', 'comment')
    toxicity_type='personalized_toxicity',  # Type of toxicity analysis to perform ('personalized_toxicity' or 'hatespeech')
    context_info=None,  # Optional: additional context about the conversation or situation
    pipeline_config=pipeline_config,  # Configuration specifying model, paths, and behavior
    serialize_result=True,  # If True, saves the result to disk as YAML
)

# Display the analysis result and toxicity verdict
print(result.answer['contains_toxicity'])
```

We also provide an [example notebook](https://github.com/debatelab/toxicity-detector/blob/main/toxicity_pipeline_intro.ipynb) that demonstrates how to run the toxicity detection pipeline with a Hugging Face API key.

## 🧭 Using the Gradio Demoapp

The project includes a Gradio web interface for interactive toxicity detection.

### Using the CLI

Run the app using the simple command:

```bash
# With app configuration file
toxicity-detector app --app-config ./config/app_config.yaml

# With pipeline configuration file (uses default app settings)
toxicity-detector app --pipeline-config ./config/pipeline_config.yaml

# With custom server settings
toxicity-detector app \
  --app-config ./config/app_config.yaml \
  --server-port 8080 \
  --share
```

The app will start and be accessible at `http://localhost:7860` by default (or your specified port).

### Configuration 

To enable developer mode with additional configuration options, update your `config/app_config.yaml`:

```yaml
developer_mode: true
```

*Note:* the configuration tab is only shown when `developer_mode: true`. If `force_agreement: true`, you must accept the agreement first.

Additional information about the different settings can be found in the [`config/app_config.yaml`](https://github.com/debatelab/toxicity-detector/blob/main/config/app_config.yaml).


## 🛠️ Configuration of the Pipeline

The pipeline is configured via a YAML file that is loaded into the Pydantic model `PipelineConfig`.

- Config schema/model: `src/toxicity_detector/config.py` (`class PipelineConfig`)
- Main entry point: `src/toxicity_detector/backend.py` (`detect_toxicity(...)`)

Key sections in [`config/pipeline_config.yaml`](https://github.com/debatelab/toxicity-detector/blob/main/config/pipeline_config.yaml):

- **Model selection**: `used_chat_model` and the `models:` dictionary (provider/model/base_url + `api_key_name`)
- **Storage**: `local_serialization`, `local_base_path`, `result_data_path`, `log_path`, `subdirectory_construction`
- **Toxicity definitions**: `toxicities:` (currently `personalized_toxicity` and `hatespeech`)
  - Each toxicity type contains `tasks:` which includes
    - `prepatory_analysis.general_questions`
    - `indicator_analysis.*` (your indicator list)
- **Prompts**: prompt templates are configurable (see `prompt_templates` in the [default pipeline config](https://github.com/debatelab/toxicity-detector/blob/main/src/toxicity_detector/package_data/default_pipeline_config.yaml))

If you want to start from a known-good baseline, the package contains a default pipeline config with all default prompts here:
[`src/toxicity_detector/package_data/default_pipeline_config.yaml`](https://github.com/debatelab/toxicity-detector/blob/main/src/toxicity_detector/package_data/default_pipeline_config.yaml).

Additional information about the different settings can be found in the [`config/pipeline_config.yaml`](https://github.com/debatelab/toxicity-detector/blob/main/config/pipeline_config.yaml).

## 🔧 Development

### Project Structure

High-level overview of the repository layout:

```
toxicity-detector/
├── config/                          # Configuration template files
│   ├── app_config.yaml              # Gradio app configuration (AppConfig)
│   └── pipeline_config.yaml         # Pipeline configuration (PipelineConfig)
├── src/
│   └── toxicity_detector/
│       ├── __init__.py
│       ├── app/                     # Gradio web interface (modularized)
│       │   ├── app.py
│       │   ├── app_config_loader.py
│       │   ├── agreement_tab.py
│       │   ├── config_tab.py
│       │   └── detection_tab.py
│       ├── backend.py               # Core detection logic (detect_toxicity)
│       ├── chains.py                # LangChain pipelines
│       ├── cli.py                   # CLI entry point (toxicity-detector)
│       ├── config.py                # Pydantic config models
│       └── managers/                # Config and persistence utilities
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

### Setup 

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

#### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

#### Installation

1. **Install uv** (if not already installed):

2. **Clone the repository**:
   ```bash
   git clone https://github.com/debatelab/toxicity-detector.git
   cd toxicity-detector
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all dependencies specified in `pyproject.toml`. If a `uv.lock` is present, `uv` will reproduce the environment specified in that file. If you want to start with a fresh environment and/or use other package versions, remove or update the `uv.lock` accordingly.

4. **Install development dependencies** (optional):
   ```bash
   uv sync --group dev
   ```

### Running Tests

Run all tests:
```bash
uv run pytest
```

Run tests with verbose output:
```bash
uv run pytest -v
```

Run a specific test file:
```bash
uv run pytest tests/test_config.py
```

Run tests with coverage report:
```bash
uv run pytest --cov=src/toxicity_detector
```

Alternative: Using the activated virtual environment:
```bash
# Activate the virtual environment first
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Then run pytest directly
pytest tests/
pytest tests/test_config.py -v
```

### Working with Notebooks

To use Jupyter notebooks for development:

```bash
# Install dev dependencies if not already done
uv sync --group dev

# Start Jupyter
uv run jupyter notebook notebooks/
```

## 🙏 Acknowledgements

### 🛠️ Powered By

- **[LangChain](https://www.langchain.com/)**: Workflow orchestration
- **[Gradio](https://gradio.app/)**: Interactive web interface
- **[Pydantic](https://pydantic.dev/)**: Data validation and configuration management
- **[Hugging Face](https://huggingface.co/)**: Model hosting and deployment

### 🏛️ Funding 

The Toxicity Detector was implemented as part of the project "Opportunities of AI to Strengthen Our Deliberative Culture" ([KIdeKu](https://compphil2mmae.github.io/research/kideku/)) which was funded by the *Federal Ministry of Education, Family Affairs, Senior Citizens, Women and Youth ([BMBFSFJ](https://www.bmbfsfj.bund.de/bmbfsfj/meta/en))*.

<a href="https://www.bmbfsfj.bund.de/bmbfsfj/meta/en">
  <img src="./img/funding.png" alt="BMFSFJ Funding" width="40%">
</a>


## 📄 License

This project is licensed under the MIT License. See `LICENSE`.
