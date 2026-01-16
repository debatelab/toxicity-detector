"""Main Gradio app module for toxicity detection."""

import gradio as gr
from loguru import logger

from toxicity_detector.result import ToxicityDetectorResult
from toxicity_detector.app.app_config_loader import (
    init_app_config,
    _app_config,
    _config_manager,
)
from toxicity_detector.app.detection_tab import (
    create_detection_tab,
    _load_toxicity_example_data,
)
from toxicity_detector.app.config_tab import create_config_tab
from toxicity_detector.app.agreement_tab import create_agreement_tab


def create_demo() -> gr.Blocks:
    """Create and return the Gradio demo interface."""
    with gr.Blocks(title="Chatbot Detektor für toxische Sprache") as demo:

        gr.Markdown(_app_config().ui_texts.app_head)

        # TODO: Refactor app to use a UI state class
        # Initialize state variables
        tw_approved_value = False if _app_config().force_agreement else True
        tw_approved = gr.State(tw_approved_value)
        result_state = gr.State(ToxicityDetectorResult())
        pipeline_config_state = gr.State(
            _config_manager().get_default_pipeline_config()
        )
        feedback_interactive_st = gr.State(False)
        output_dirty_st = gr.State(True)
        feedback_likert_content_st = gr.State(dict())
        toxicity_example_data = _load_toxicity_example_data()
        toxicity_example_data_st = gr.State(toxicity_example_data)
        user_input_source_st = gr.State("")

        with gr.Tabs(
            selected="detector_tab" if tw_approved_value else "tw_tab"
        ) as tabs:
            # Create tabs
            detector_tab = create_detection_tab(
                tw_approved=tw_approved,
                tw_approved_value=tw_approved_value,
                pipeline_config_state=pipeline_config_state,
                result_state=result_state,
                feedback_interactive_st=feedback_interactive_st,
                output_dirty_st=output_dirty_st,
                feedback_likert_content_st=feedback_likert_content_st,
                toxicity_example_data_st=toxicity_example_data_st,
                user_input_source_st=user_input_source_st,
            )

            config_tab = create_config_tab(
                tw_approved=tw_approved,
                tw_approved_value=tw_approved_value,
                pipeline_config_state=pipeline_config_state,
            )

            tw_tab = create_agreement_tab(
                tw_approved=tw_approved,
                detector_tab=detector_tab,
                config_tab=config_tab,
                tabs=tabs,
            )

    return demo


def launch_app(
    config_path=None,
    config_type="app",
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
):
    """Launch the Gradio app with specified configuration.

    Args:
        config_path: Path to configuration file. If None, uses default.
        config_type: Either "app" or "pipeline". Determines how to load config.
        server_name: Server name/host (default: 127.0.0.1)
        server_port: Server port (default: 7860)
        share: Whether to create a public shareable link (default: False)
    """
    # Initialize app config
    logger.info(f"Loading {config_type} config from: {config_path}")
    init_app_config(config_path, config_type)

    # Create the Gradio demo
    demo = create_demo()

    # Launch the server
    logger.info(f"Starting Gradio app on {server_name}:{server_port}")
    demo.launch(
        show_error=True,
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
