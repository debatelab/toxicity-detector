import gradio as gr
import yaml
import os
import pandas as pd
from typing import Dict, List, Tuple
from gradio_highlightedcode import HighlightedCode
from random import randrange
from loguru import logger

# Import from toxicity_detector package
from toxicity_detector import (
    detect_toxicity,
    get_toxicity_example_data,
    dump_pipeline_config_str,
    config_file_exists,
    pipeline_config_as_string,
    pipeline_config_file_names,
    update_feedback,
)
from toxicity_detector.config import AppConfig, PipelineConfig
from toxicity_detector.result import ToxicityDetectorResult

# Load app config file path from environment variable
_APP_CONFIG_FILE = os.getenv(
    "TOXICITY_DETECTOR_APP_CONFIG_FILE", "./config/app_config.yaml"
)

# Global Inits

# loading app config as dict from yaml
app_config = AppConfig.from_file(_APP_CONFIG_FILE)

config_file_names = pipeline_config_file_names(app_config)
logger.info(f"Valid configs: {config_file_names}")

# variable is set on HF via the space
if "RUNS_ON_SPACES" not in os.environ.keys():
    logger.info("Gradioapp runs locally. Loading env variables...")
    from dotenv import load_dotenv

    load_dotenv()

# HELPER FUNCTIONS


def _tasks(pipeline_config: PipelineConfig, toxicity_type) -> List[str]:
    task_names = []
    task_groups = pipeline_config.toxicities[toxicity_type].tasks.keys()
    for task_group in task_groups:
        task_names.extend(
            list(pipeline_config.toxicities[toxicity_type].tasks[task_group].keys())
        )
    return task_names


def _load_toxicity_example_data(
    app_config: AppConfig
) -> pd.DataFrame:
    examples_data_file = None
    if app_config.toxicity_examples_data_file is not None:
        examples_data_file = app_config.toxicity_examples_data_file
        msg = (
            "Loading toxicity examples as specified in app config "
            f"({examples_data_file})"
        )
        logger.info(msg)
    else:
        logger.warning(
            "No toxicity example data file specified in app config! "
        )

    return get_toxicity_example_data(app_config, examples_data_file)


with gr.Blocks(title="Chatbot Detektor für toxische Sprache") as demo:

    gr.Markdown(app_config.ui_texts.app_head)

    tw_approved = gr.State(False)
    # uuid variable for the detection request
    # (used for UI logic to attach user feedback)
    result_state = gr.State(ToxicityDetectorResult())
    pipeline_config_state = gr.State(app_config.get_default_pipeline_config())

    # state variable to control the interactivity of the feedback elements
    feedback_interactive_st = gr.State(False)
    # state variable to set output elements as dirty
    output_dirty_st = gr.State(True)
    # state variable to store the feedback from the likert scales
    # (feedback radio buttons)
    feedback_likert_content_st = gr.State(dict())
    # state variable to store the example data for the toxicity detection
    toxicity_example_data_st = gr.State(
        _load_toxicity_example_data(app_config)
    )

    # state variable to store source string for the user input
    user_input_source_st = gr.State("")
    with gr.Tabs(
        selected="detector_tab" if app_config.developer_mode else "tw_tab"
    ) as tabs:
        # TAB: TOXICITY DETECTION
        with gr.Tab(
            label="Toxizitätsdetektor",
            id="detector_tab",
            visible=True if app_config.developer_mode else False,
        ) as detector_tab:
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    init_toxicity_key = list(
                        pipeline_config_state.value.toxicities.keys()
                    )[0]
                    radio_toxicitiy_type = gr.Radio(
                        [
                            (value.title, key)
                            for (
                                key,
                                value,
                            ) in pipeline_config_state.value.toxicities.items()
                        ],
                        value=init_toxicity_key,
                        label="Toxizitätsdefinition",
                        info=("Welche Art von Toxizität soll " "detektiert werden?"),
                    )
                    with gr.Accordion("Definition der gewählten Toxizitätsart:"):
                        md_toxicity_description = gr.Markdown(
                            pipeline_config_state.value.toxicities[
                                init_toxicity_key
                            ].user_description
                        )

                    # TODO: allow model selection if set in config
                    # dropdown_model = gr.Dropdown(
                    #         [(value['name'], key)
                    #          for (key, value)
                    #          in pipeline_config_dict.value['models'].items()
                    #          ],
                    #         value=list(pipeline_config_dict.value['models'].keys())[0],
                    #         label="Benutztes Sprachmodell",
                    #         info="Wähle das zu benutzende Modell aus!"
                    # )
                    user_input_tb = gr.Textbox(
                        label="Texteingabe",
                        info="Eingabe des zu kategorisierenden Textes.",
                    )
                    random_example_btn = gr.Button("Zufälliges Beispiel")
                    context_tb = gr.Textbox(
                        label="Kontextinfo",
                        info=("Eingabe von Kontextinformation " "(kann leer bleiben)"),
                    )
                    categorize_btn = gr.Button("Detect Toxicity")

                with gr.Column(scale=2, min_width=300):
                    general_questions_output_tb = gr.Textbox(
                        label=(
                            "General questions output/preprocessing " "(developer mode)"
                        ),
                        visible=(True if app_config.developer_mode else False),
                    )
                    indicators_output_tb = gr.Textbox(
                        label="Indicator analysis output (developer mode)",
                        visible=(True if app_config.developer_mode else False),
                    )
                    ouput_text_box = gr.Textbox(
                        label="Kategorisierung der Eingabe durch den Detektor"
                    )
                    feedback_radio = gr.Radio(
                        [
                            (value, key)
                            for (key, value) in app_config.feedback[
                                "likert_scale"
                            ].items()
                        ],
                        label="Korrektheit der Kategorisierung",
                        info=(
                            "Stimmt die Kategorisierung des Detekors? "
                            "(Bist Du dir selbst unsicher, ob die Eingabe "
                            "toxischen Inhalt enthält, kreuze 'unklar' an.)"
                        ),
                        interactive=False,
                    )
                    feedback_textbox = gr.Textbox(
                        label="Feedback:",
                        info=(
                            "Hier kannst Du ausführliches Feedback zur "
                            "Kategoriesung des Textes durch den Detektor "
                            "eingeben."
                        ),
                        interactive=False,
                    )
                    with gr.Accordion(
                        "Taskspecific feedback (developer mode)",
                        visible=(True if app_config.developer_mode else False),
                    ):

                        @gr.render(
                            inputs=[
                                radio_toxicitiy_type,
                                feedback_interactive_st,
                            ]
                        )
                        def show_indicator_feedback_radios(
                            toxicity_type: str,
                            interactive: bool,
                        ):
                            global app_config
                            for task in _tasks(
                                pipeline_config_state.value, toxicity_type
                            ):
                                radio = gr.Radio(
                                    [
                                        (value, key)
                                        for (
                                            key,
                                            value,
                                        ) in app_config.feedback["likert_scale"].items()
                                    ],
                                    label=(
                                        f"Korrektheit der Antwort "
                                        f"(Indikator: {task})"
                                    ),
                                    info=(
                                        "Stimmt die Antwort/Beschreibung des "
                                        "Detekors? (Bist Du dir selbst "
                                        "unsicher, was eine korrekt Antwort "
                                        "ist, kreuze 'unklar' an.)"
                                    ),
                                    interactive=interactive,
                                    value=None,
                                )

                                def update_indicator_feedback(
                                    task: str,
                                    indicator_feedback: str,
                                    feedback_likert_content: Dict,
                                ):
                                    if indicator_feedback:
                                        feedback_likert_content[task] = (
                                            indicator_feedback
                                        )
                                    return feedback_likert_content

                                # event listener for the radio button
                                # (to update the feedback content)
                                radio.change(
                                    lambda indicator_feedback, feedback_likert_content, task=task: update_indicator_feedback(  # noqa: E501
                                        task,
                                        indicator_feedback,
                                        feedback_likert_content,
                                    ),
                                    [radio, feedback_likert_content_st],
                                    [feedback_likert_content_st],
                                )

                    feedback_btn = gr.Button(
                        "Feedback speichern/aktualisieren", interactive=False
                    )

            # EVENT LISTENER/LOGIC FOR DETECTION TAB
            def random_input_example(
                toxicity_example_data: pd.DataFrame,
            ) -> Tuple[str, str]:
                example = toxicity_example_data.loc[
                    randrange(len(toxicity_example_data)), :
                ]
                return (str(example["text"]), str(example["source"]))

            random_example_btn.click(
                random_input_example,
                toxicity_example_data_st,
                [user_input_tb, user_input_source_st],
            )
            # set output dirty when changing input
            user_input_tb.change(lambda: True, None, output_dirty_st)
            # set input source string if user edits the input
            user_input_tb.input(
                lambda: "kideku_toxicity_detector", None, user_input_source_st
            )

            # if changed to dirty, we clear the output textboxes and
            # deactivate the feedback ui
            output_dirty_st.change(
                lambda dirty: (
                    (
                        gr.Textbox(interactive=False, value="")
                        if dirty
                        else gr.Textbox(interactive=False)
                    ),
                    (
                        gr.Textbox(interactive=False, value="")
                        if dirty
                        else gr.Textbox(interactive=False)
                    ),
                    (
                        gr.Textbox(interactive=False, value="")
                        if dirty
                        else gr.Textbox(interactive=False)
                    ),
                    not dirty,  # interactive feedback ui
                    dict(),  # feedback content
                ),
                output_dirty_st,
                [
                    ouput_text_box,
                    general_questions_output_tb,
                    indicators_output_tb,
                    feedback_interactive_st,
                    feedback_likert_content_st,
                ],
            )
            # de-/activation of feedback ui
            feedback_interactive_st.change(
                lambda interactive: (
                    gr.Radio(interactive=interactive, value=None),
                    gr.Textbox(interactive=interactive, value=""),
                    gr.Button(interactive=interactive),
                ),
                feedback_interactive_st,
                [feedback_radio, feedback_textbox, feedback_btn],
            )

            # Detection button
            def detect_toxicity_wrapper(
                input_text: str,
                user_input_source: str,
                toxicity_type: str,
                context_info: str,
                pipeline_config: PipelineConfig,
            ):

                result = detect_toxicity(
                    input_text=input_text,
                    user_input_source=user_input_source,
                    toxicity_type=toxicity_type,
                    context_info=context_info,
                    pipeline_config=pipeline_config,
                )

                indicator_result = result.answer["indicator_analysis"]
                # indicator analysis as one string for the ouput
                indicator_analysis_str = "".join(
                    [
                        "".join([key, ": ", value, "\n\n"])
                        for key, value in indicator_result.items()
                    ]
                )

                return (
                    result.answer["analysis_result"],
                    result.answer[
                        "preprocessing_results"
                    ],  # ouput for text field (dev mode)
                    indicator_analysis_str,  # output for textfield (dev mode)
                    # feedback ui interactive via `feedback_interactive_st`
                    True,
                    dict(),  # feedback content
                    False,  # output dirty
                    result,
                )

            categorize_btn.click(
                fn=detect_toxicity_wrapper,
                inputs=[
                    user_input_tb,
                    user_input_source_st,
                    radio_toxicitiy_type,
                    context_tb,
                    pipeline_config_state,
                ],
                outputs=[
                    ouput_text_box,
                    general_questions_output_tb,
                    indicators_output_tb,
                    feedback_interactive_st,
                    feedback_likert_content_st,
                    output_dirty_st,
                    result_state,
                ],
            )
            # Changing toxicity type: -> update description
            # and set output uis to dirty
            radio_toxicitiy_type.change(
                lambda toxicity_type, pipeline_config: (
                    pipeline_config.toxicities[toxicity_type].user_description,
                    True,
                ),
                [radio_toxicitiy_type, pipeline_config_state],
                [md_toxicity_description, output_dirty_st],
            )

            # Saving feedback
            feedback_btn.click(
                lambda v, w, x, y, z: update_feedback(v, w, x, y, z),
                [
                    pipeline_config_state,
                    result_state,
                    feedback_textbox,
                    feedback_radio,
                    feedback_likert_content_st,
                ],
                None,
            )

            # UPDATE UI ELEMENTS IF MODEL_CONFIG CHANGES
            # update toxicity description, reload example data and
            # set output ui elements to dirty
            pipeline_config_state.change(
                lambda toxicity_type, pipeline_config: (
                    pipeline_config.toxicities[toxicity_type].user_description,
                    _load_toxicity_example_data(app_config),
                    ToxicityDetectorResult(),
                    True,  # set output ui elements to dirty
                ),
                [radio_toxicitiy_type, pipeline_config_state],
                [
                    md_toxicity_description,
                    toxicity_example_data_st,
                    result_state,
                    output_dirty_st,
                ],
                show_progress="hidden",
            )

        # TAB: CONFIGUARTION
        with gr.Tab(
            label="Konfiguration",
            id="config_tab",
            visible=True if app_config.developer_mode else False,
        ) as config_tab:
            # with gr.Tab(label="Konfiguration", id="config_tab",
            #             visible=False) as config_tab:
            with gr.Row():
                with gr.Column(scale=4, min_width=300):
                    # default pipeline config as str
                    pipeline_config_str = pipeline_config_as_string(app_config)
                    yaml_config_input = HighlightedCode(
                        pipeline_config_str, language="yaml", interactive=True
                    )
                with gr.Column(scale=1, min_width=50):
                    dropdown_config = gr.Dropdown(
                        choices=config_file_names,
                        value=os.path.basename(app_config.default_pipeline_config_file),
                        allow_custom_value=False,
                        label="Konfigurationsdatei",
                        info="Wähle die zu ladende Konfigurationsdatei aus!",
                        interactive=True,
                    )
                    reload_config_btn = gr.Button("Eingegebene Konfiguration laden")
                    with gr.Group():
                        gr.Markdown("  Speichern der aktuellen Konfiguration.")
                        new_config_name_tb = gr.Textbox(
                            label="Name (ohne Dateiendung):"
                        )
                        save_config_btn = gr.Button("Speichern")

            # EVENT LISTENER/LOGIC FOR CONFIG TAB
            def parse_yaml_str(yaml_str: str):
                try:
                    config_dict = yaml.safe_load(yaml_str)
                    return config_dict
                except yaml.YAMLError as e:
                    raise gr.Error(f"Error parsing YAML: {e}")

            reload_config_btn.click(
                lambda yaml_str: PipelineConfig(**parse_yaml_str(yaml_str)),
                yaml_config_input,
                pipeline_config_state,
            )

            def load_selected_config(
                config_file_name: str,
            ) -> Tuple[HighlightedCode, PipelineConfig, gr.Dropdown]:
                global config_file_names
                config_path = os.path.join(
                    app_config.get_pipeline_config_path(), config_file_name
                )
                if app_config.local_pipeline_config:
                    pipeline_config = PipelineConfig.from_file(config_path)
                else:
                    pipeline_config = PipelineConfig.from_hf(config_path)
                pipeline_config_str = pipeline_config_as_string(
                    app_config, config_file_name
                )
                yaml_input_str = HighlightedCode(
                    pipeline_config_str, language="yaml", interactive=True
                )
                dropdown_config = gr.Dropdown(
                    choices=config_file_names,
                    value=config_file_name,
                    allow_custom_value=False,
                    label="Konfigurationsdatei",
                    info="Wähle die zu ladende Konfigurationsdatei aus!",
                )
                return yaml_input_str, pipeline_config, dropdown_config

            dropdown_config.input(
                load_selected_config,
                dropdown_config,
                [
                    yaml_config_input,
                    pipeline_config_state,
                    dropdown_config,
                ],  # update config text field and config_dict
                show_progress="full",
            )

            # SAVE-PIPELINE-CONFIG BUTTON
            def save_config(new_config_name: str, config_str: str):
                global config_file_names
                if not new_config_name or new_config_name.isspace():
                    raise gr.Error(
                        "Der Name der neuen Konfiguration " "darf nicht leer sein."
                    )
                # save str from yaml_config_input as file
                new_config_file_name = f"{new_config_name}.yaml"
                if config_file_exists(app_config, new_config_file_name):
                    raise gr.Error(
                        f"Eine Konfigurationsdatei mit dem Name "
                        f"{new_config_file_name} existiert schon."
                    )

                dump_pipeline_config_str(new_config_file_name, config_str, app_config)

                # Update the dropdown with the new config file
                config_file_names.append(new_config_file_name)

                return (
                    gr.Dropdown(
                        choices=config_file_names,
                        value=new_config_file_name,
                        interactive=True,
                    ),
                    PipelineConfig(**parse_yaml_str(config_str)),
                )

            save_config_btn.click(
                save_config,
                [new_config_name_tb, yaml_config_input],
                [dropdown_config, pipeline_config_state],
            )
        # TAB AGREEMENT
        with gr.Tab(
            label="Benutzungshinweise",
            id="tw_tab",
            visible=False if app_config.developer_mode else True,
        ) as tw_tab:

            gr.Markdown(app_config.ui_texts.trigger_warning["message"])
            tw_checkbox = gr.Checkbox(
                label=app_config.ui_texts.trigger_warning["checkbox_label"]
            )
            tw_checkbox.input(
                lambda x: (
                    x,
                    gr.Checkbox(
                        label=app_config.ui_texts.trigger_warning["checkbox_label"],
                        interactive=False,
                    ),
                    gr.Tab("Toxizitätsdetektor", visible=True),
                    (
                        gr.Tab("Konfiguration", visible=True)
                        if app_config.developer_mode
                        else gr.Tab("Konfiguration", visible=False)
                    ),
                    gr.Tabs(selected="detector_tab"),
                ),
                tw_checkbox,
                [tw_approved, tw_checkbox, detector_tab, config_tab, tabs],
            )

if __name__ == "__main__":
    demo.launch(show_error=True)
