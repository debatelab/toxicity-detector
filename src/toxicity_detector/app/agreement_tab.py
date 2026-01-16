"""Agreement tab for the Gradio app."""

import gradio as gr

from toxicity_detector.app.app_config_loader import _app_config


def create_agreement_tab(
    tw_approved: gr.State,
    detector_tab: gr.Tab,
    config_tab: gr.Tab,
    tabs: gr.Tabs,
) -> gr.Tab:
    """Create the user agreement/trigger warning tab.
    
    Args:
        tw_approved: State for trigger warning approval
        detector_tab: The detector tab component
        config_tab: The config tab component
        tabs: The tabs container
        
    Returns:
        The agreement tab component
    """
    with gr.Tab(
        label="Benutzungshinweise",
        id="tw_tab",
        visible=_app_config().force_agreement,
    ) as tw_tab:
        gr.Markdown(_app_config().ui_texts.trigger_warning["message"])
        tw_checkbox = gr.Checkbox(
            label=_app_config().ui_texts.trigger_warning["checkbox_label"]
        )
        tw_checkbox.input(
            lambda x: (
                x,
                gr.Checkbox(interactive=False),
                gr.Tab(visible=x),
                gr.Tab(visible=x and _app_config().developer_mode),
                gr.Tabs(selected="detector_tab" if x else "tw_tab"),
            ),
            tw_checkbox,
            [tw_approved, tw_checkbox, detector_tab, config_tab, tabs],
        )
    
    return tw_tab
