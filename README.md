# Toxicity Detector

An LLM-based pipeline to detect toxic speech using language models.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

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

   This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

4. **Install development dependencies** (optional):
   ```bash
   uv sync --group dev
   ```

### Environment Variables

Create a `.env` file in the project root with the following variables:

```txt
# API Keys (by the names as specified in the model config files)

# Optional: Custom app config file path
TOXICITY_DETECTOR_APP_CONFIG_FILE=./config/app_config.yaml
```
## Running the Pipeline 

See the notebooks in the `notebooks/` directory for examples on how to run and test the toxicity detection pipeline.

## Running the Gradio App

The project includes a Gradio web interface for interactive toxicity detection.

### Basic Usage

Run the app using uv:

```bash
uv run python src/toxicity_detector/app.py
```

The app will start and be accessible at `http://localhost:7860` by default.

### Alternative: Using the uv shell

You can also activate the virtual environment and run the app directly:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Run the app
python src/toxicity_detector/app.py

# or (enables live reloading)
gradio src/toxicity_detector/app.py
```

### Developer Mode

To enable developer mode with additional configuration options, update your `config/app_config.yaml`:

```yaml
developer_mode: true
```

## Project Structure

```
toxicity-detector/
├── config/                          # Configuration files
│   ├── app_config.yaml             # App configuration
│   └── default_model_config_*.yaml # Model configurations
├── src/
│   └── toxicity_detector/
│       ├── __init__.py
│       ├── app.py                  # Gradio web interface
│       ├── backend.py              # Core detection logic
│       └── chains.py               # LangChain pipelines
├── logs/                           # Application logs
├── notebooks/                      # Jupyter notebooks for testing
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

## Development

### Code Style

The project follows PEP 8 guidelines with a maximum line length of 88 characters.

Run linting checks:
```bash
uv run flake8 src/
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

## License

See [LICENSE](LICENSE) file for details.
