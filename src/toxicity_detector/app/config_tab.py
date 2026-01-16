"""Configuration tab for the Gradio app."""

import os
import gradio as gr
import yaml
from typing import Tuple
from loguru import logger

from toxicity_detector.config import PipelineConfig
from toxicity_detector.app.app_config_loader import _app_config, _config_manager

# TODO: Use decorators consistently for event handling.


def create_config_tab(
    tw_approved: gr.State,
    tw_approved_value: bool,
    pipeline_config_state: gr.State,
) -> gr.Tab:
    """Create the configuration tab.

    Args:
        tw_approved: State for trigger warning approval
        tw_approved_value: Boolean value indicating if trigger warning was approved
        pipeline_config_state: State for pipeline configuration

    Returns:
        The configuration tab component
    """
    with gr.Tab(
        label="Konfiguration",
        id="config_tab",
        visible=tw_approved_value and _app_config().developer_mode,
    ) as config_tab:
        with gr.Row():
            with gr.Column(scale=4, min_width=300):
                yaml_config_input = gr.Code(
                    _config_manager().get_pipeline_config_as_string(),
                    language="yaml",
                    interactive=True,
                )
            with gr.Column(scale=1, min_width=50):
                dropdown_config = gr.Dropdown(
                    choices=_config_manager().list_config_files(),
                    value=os.path.basename(
                        _config_manager().default_pipeline_config_file
                    ),
                    allow_custom_value=False,
                    label="Konfigurationsdatei",
                    info="Wähle die zu ladende Konfigurationsdatei aus!",
                    interactive=True,
                )
                reload_config_btn = gr.Button("Eingegebene Konfiguration laden")
                with gr.Group():
                    gr.Markdown("  Speichern der aktuellen Konfiguration.")
                    new_config_name_tb = gr.Textbox(
                        label="Dateiname (mit Dateiendung .yaml)",
                    )
                    save_config_btn = gr.Button("Speichern")

        # EVENT LISTENER/LOGIC FOR CONFIG TAB
        @reload_config_btn.click(  # RELOAD CONFIG BUTTON
            inputs=yaml_config_input, outputs=pipeline_config_state
        )
        def parse_yaml_str(yaml_str: str) -> PipelineConfig:
            logger.debug("Loading pipeline config from YAML string...")
            try:
                config_dict = yaml.safe_load(yaml_str)
                return PipelineConfig(**config_dict)
            except yaml.YAMLError as e:
                raise gr.Error(f"Error parsing YAML: {e}")

        @dropdown_config.input(
            inputs=dropdown_config,
            outputs=[yaml_config_input, pipeline_config_state],
            show_progress="minimal",
        )
        def load_selected_config(  # type: ignore
            config_file_name: str,
        ) -> Tuple[str, PipelineConfig]:
            logger.info(f"Loading selected config: {config_file_name}")
            pipeline_config = _config_manager().load_pipeline_config(config_file_name)
            pipeline_config_str = _config_manager().get_pipeline_config_as_string(
                config_file_name
            )
            return pipeline_config_str, pipeline_config

        # SAVE-PIPELINE-CONFIG BUTTON
        @save_config_btn.click(
            inputs=[new_config_name_tb, yaml_config_input],
            outputs=[dropdown_config, pipeline_config_state],
        )
        def save_config(new_config_name: str, config_str: str):
            if not new_config_name or new_config_name.isspace():
                raise gr.Error(
                    "Der Name der neuen Konfiguration " "darf nicht leer sein."
                )
            try:
                logger.debug("Loading pipeline config from YAML string...")
                config = PipelineConfig(**yaml.safe_load(config_str))
            except yaml.YAMLError as e:
                raise gr.Error(f"Error parsing YAML: {e}")
            # save str from yaml_config_input as file
            new_config_file_name = f"{new_config_name}.yaml"
            if _config_manager().file_exists(new_config_file_name):
                raise gr.Error(
                    f"Eine Konfigurationsdatei mit dem Name "
                    f"{new_config_file_name} existiert schon."
                )
            _config_manager().save_string(new_config_name, config_str)
            # Update the dropdown with the new config file
            return (
                gr.Dropdown(
                    choices=_config_manager().list_config_files(),
                    value=new_config_file_name,
                    interactive=True,
                ),
                config,
            )

    return config_tab
