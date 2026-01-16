"""Detection tab for the Gradio app."""

import gradio as gr
import pandas as pd
from typing import Dict, Tuple
from random import randrange

from toxicity_detector import detect_toxicity, update_feedback
from toxicity_detector.config import PipelineConfig
from toxicity_detector.result import ToxicityDetectorResult
from toxicity_detector.app.app_config_loader import _app_config, _config_manager


def _tasks(pipeline_config: PipelineConfig, toxicity_type) -> list[str]:
    """Get list of task names for a toxicity type."""
    task_names = []
    task_groups = pipeline_config.toxicities[toxicity_type].tasks.keys()
    for task_group in task_groups:
        task_names.extend(
            list(pipeline_config.toxicities[toxicity_type].tasks[task_group].keys())
        )
    return task_names


def _load_toxicity_example_data() -> pd.DataFrame | None:
    """Load toxicity example data if configured."""
    if (data_file := _app_config().toxicity_examples_data_file) is not None:
        return _config_manager().load_toxicity_example_data(data_file)
    else:
        return None

# TODO: Use decorators consistently for event handling.

def create_detection_tab(
    tw_approved: gr.State,
    tw_approved_value: bool,
    pipeline_config_state: gr.State,
    result_state: gr.State,
    feedback_interactive_st: gr.State,
    output_dirty_st: gr.State,
    feedback_likert_content_st: gr.State,
    toxicity_example_data_st: gr.State,
    user_input_source_st: gr.State,
) -> gr.Tab:
    """Create the toxicity detection tab.

    Args:
        tw_approved: State for trigger warning approval
        tw_approved_value: Boolean value indicating if trigger warning was approved
        pipeline_config_state: State for pipeline configuration
        result_state: State for detection results
        feedback_interactive_st: State for feedback UI interactivity
        output_dirty_st: State for output dirty flag
        feedback_likert_content_st: State for feedback content
        toxicity_example_data_st: State for example data
        user_input_source_st: State for input source string

    Returns:
        The detection tab component
    """
    with gr.Tab(
        label="Toxizitätsdetektor",
        id="detector_tab",
        visible=tw_approved_value,
    ) as detector_tab:
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                init_toxicity_key = list(pipeline_config_state.value.toxicities.keys())[
                    0
                ]
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

                user_input_tb = gr.Textbox(
                    label="Texteingabe",
                    info="Eingabe des zu kategorisierenden Textes.",
                    lines=6,
                )
                random_example_btn = gr.Button(
                    "Zufälliges Beispiel",
                    interactive=toxicity_example_data_st.value is not None,
                )
                with gr.Accordion(label="Kontextinfo (kann leer bleiben)", open=False):
                    context_tb = gr.Textbox(
                        lines=6,
                        container=False,
                    )
                categorize_btn = gr.Button("Detect Toxicity")

            with gr.Column(scale=2, min_width=300):
                with gr.Accordion(
                    label="Zwischenergebnisse der Pipeline (developer mode)",
                    visible=(True if _app_config().developer_mode else False),
                    open=False,
                ):
                    general_questions_output_tb = gr.Textbox(
                        label=("General questions output/preprocessing "),
                        visible=(True if _app_config().developer_mode else False),
                        lines=10,
                    )
                    indicators_output_tb = gr.Textbox(
                        label="Indicator analysis output (developer mode)",
                        visible=(True if _app_config().developer_mode else False),
                        lines=10,
                    )
                with gr.Accordion(
                    label="Ergebnis der Toxizitätskategorisierung", open=True
                ):
                    output_text_box = gr.Textbox(
                        label="Kategorisierung der Eingabe durch den Detektor",
                        lines=10,
                    )
                    output_label_text_box = gr.Textbox(
                        label="Label der Kategorisierung",
                        info="true/false/unclear ('unclear' = unsicher/unklar/nicht eindeutig)",
                    )
                feedback_radio = gr.Radio(
                    [
                        (value, key)
                        for (key, value) in _app_config()
                        .feedback["likert_scale"]
                        .items()
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
                    visible=(True if _app_config().developer_mode else False),
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
                        for task in _tasks(pipeline_config_state.value, toxicity_type):
                            radio = gr.Radio(
                                [
                                    (value, key)
                                    for (
                                        key,
                                        value,
                                    ) in _app_config()
                                    .feedback["likert_scale"]
                                    .items()
                                ],
                                label=(
                                    f"Korrektheit der Antwort " f"(Indikator: {task})"
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
                                    feedback_likert_content[task] = indicator_feedback
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
            toxicity_example_data: pd.DataFrame | None,
        ) -> Tuple[str, str]:
            if toxicity_example_data is None or len(toxicity_example_data) == 0:
                return ("", "")
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
                output_text_box,
                output_label_text_box,
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
                result.answer["contains_toxicity"].value,
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
                output_text_box,
                output_label_text_box,
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
        # update toxicity description,
        # and set output ui elements to dirty
        def update_on_config_change(
            toxicity_type: str, pipeline_config: PipelineConfig
        ):
            return (
                pipeline_config.toxicities[toxicity_type].user_description,
                ToxicityDetectorResult(),
                True,  # set output ui elements to dirty
            )

        pipeline_config_state.change(
            update_on_config_change,
            [radio_toxicitiy_type, pipeline_config_state],
            [
                md_toxicity_description,
                result_state,
                output_dirty_st,
            ],
            show_progress="hidden",
        )

    return detector_tab
