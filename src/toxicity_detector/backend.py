from glob import glob
import os
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from huggingface_hub import InferenceClient
import yaml
from huggingface_hub import HfFileSystem
from datetime import datetime
import pandas as pd
from loguru import logger

from toxicity_detector import MonoModelDetectToxicityChain
from toxicity_detector.result import ToxicityDetectorResult
from toxicity_detector.config import AppConfig, SubdirConstruction, PipelineConfig


def detect_toxicity(
    input_text: str,
    user_input_source: str | None,
    toxicity_type: str,  # TODO: use enum
    context_info: str | None,
    pipeline_config: PipelineConfig,
    serialize_result: bool = True,
):
    result = ToxicityDetectorResult(
        user_input=input_text,
        user_input_source=user_input_source,
        toxicity_type=toxicity_type,  # TODO: use enum
        context_information=context_info,
        pipeline_config=pipeline_config,
    )

    if not input_text or input_text == "":
        raise ValueError("Input text must not be empty.")

    # create new uuid for the detection request
    # (used for UI logic to attach user feedback and for data serialization)

    logger.info(f"Starting new detection request (uuid: {result.request_id}).")
    context_info = None if not context_info or context_info.isspace() else context_info
    model = pipeline_config.used_chat_model

    logger.info(f"Chosen toxicity type: {toxicity_type}")
    logger.info(f"Used model: {pipeline_config.models[model]['name']}")
    logger.info(f"Kontextinfo: {context_info}")
    # Chat model
    if pipeline_config.models[model]["llm_chain"] == "chat-chain":
        # getting api key
        if "api_key" in pipeline_config.models[model].keys():
            api_key = SecretStr(pipeline_config.models[model]["api_key"])
        elif "api_key_name" in pipeline_config.models[model].keys():
            api_key_name = pipeline_config.models[model]["api_key_name"]
            logger.info(f"Used api key name: {api_key_name}")
            # check whether the api key is set as env variable
            if os.environ.get(api_key_name) is None:
                raise ValueError(
                    f"The api key name {api_key_name} is not set as " f"env variable."
                )
            api_key = SecretStr(os.environ.get(api_key_name, "no-api-key"))
        else:
            raise ValueError(
                "You should specify in the config yaml either an api key "
                "or an api-key name (if it is to be found as env variable)."
            )
        if api_key is None:
            raise ValueError(
                "You should specify an api key (recommended: as env variable)."
            )
        # model params
        model_kwargs = {}
        if "model_kwargs" in pipeline_config.models[model].keys():
            model_kwargs = pipeline_config.models[model]["model_kwargs"]

        logger.info(f"Model kwargs: {model_kwargs}")
        # building chain
        toxicitiy_detection_chain = MonoModelDetectToxicityChain.build(
            llms_dict={
                "chat_model": get_openai_chat_model(
                    api_key=api_key,
                    model=pipeline_config.models[model]["model"],
                    base_url=pipeline_config.models[model]["base_url"],
                )
            },
            indicators_dict={
                key: pipeline_config.toxicities[toxicity_type]
                .tasks["indicator_analysis"][key]
                .model_dump()
                for key in pipeline_config.toxicities[toxicity_type]
                .tasks["indicator_analysis"]
                .keys()
            },
            **model_kwargs,
        )
    else:
        err_msg = (
            f"llm_chain "
            f"{pipeline_config.models[model]['llm_chain']} not "
            f"implemented."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    # TODO: Offering possibility to use zero shot classifiers
    # (for, e.g., indicator analysis)
    # using zero-shot classifier
    # elif pipeline_config.models[model]['llm_chain'] == \
    #         "zero-shot-chain":
    #     # Identifier of the model (as used in the config)
    #     # that is being used to provide an explanation for the
    #     # zero-shot categorisation.
    #     explaining_model_name = model_config_dict['toxicities'][
    #         toxicity_type]['zero-shot-categorization'][
    #         'explaining_model']
    #     # The "chain" for justifying/explaining the categorization
    #     log_msg(model_config_dict['toxicities'][toxicity_type][
    #         'zero-shot-categorization']['labels'].values())
    #     toxicity_detection_chain = \
    #         chains.IdentifyToxicContentZeroShotChain.build({
    #         'zero_shot_model': backend.ZeroShotClassifier(
    #             model=pipeline_config.models[model]['repo_id'],
    #             labels=list(model_config_dict['toxicities'][
    #                 toxicity_type]['zero-shot-categorization'][
    #                 'labels'].values()),
    #             multi_label=model_config_dict['toxicities'][toxicity_type]['zero-shot-categorization']['multi_label'],
    #             hypothesis_template=model_config_dict['toxicities'][toxicity_type]['zero-shot-categorization']['hypothesis_template'],
    #             api_token=SecretStr(os.environ.get("HF_TOKEN_KIDEKU_INFERENCE"))
    #         ),
    #         'chat_model': backend.get_chat_model(
    #             token=os.environ.get("HF_TOKEN_KIDEKU_INFERENCE"),
    #             repo_id=pipeline_config.models[explaining_model_name]['repo_id']
    #         )
    #     })

    indicators_dict = {
        key: pipeline_config.toxicities[toxicity_type]
        .tasks["indicator_analysis"][key]
        .model_dump()
        for key in pipeline_config.toxicities[toxicity_type]
        .tasks["indicator_analysis"]
        .keys()
    }

    answer = toxicitiy_detection_chain.invoke(
        {
            "system_prompt": pipeline_config.system_prompt,
            "toxicity_explication": pipeline_config.toxicities[
                toxicity_type
            ].llm_description,
            "user_input": input_text,
            "user_input_source": user_input_source,
            "general_questions": pipeline_config.toxicities[toxicity_type]
            .tasks["prepatory_analysis"]["general_questions"]
            .model_dump(),
            "context_information": context_info,
            "indicators_dict": indicators_dict,
        }
    )

    result.answer = answer
    # adding prompts for logging
    prompts = MonoModelDetectToxicityChain.prompts(**result.answer)
    result.answer["prompts"] = prompts
    # saving the result
    # TODO: as async
    if serialize_result:
        save_result(
            result=result,
            pipeline_config=pipeline_config,
        )

    return result


class ZeroShotClassifier(LLM):
    """LLM wrapper for zero shot classification via HF API."""

    model: str
    """Model ID or deployed Endpoint URL for inference."""

    labels: List[str]
    """List of label verbalizations for input text."""

    multi_label: bool = False
    """If True, labels evaluated independently. If False, mutually
    exclusive (sum to 1)."""

    hypothesis_template: str | None = None
    """
    A template sentence string with curly brackets to which the label strings are added. The label strings are added at the position of the curly brackets ”{}“. Zero-shot classifiers are based on NLI models, which evaluate if a hypothesis is entailed in another text or not. For example, with hypothesis_template=“This text is about {}.” and labels=[“economics”, “politics”], the system internally creates the two hypotheses “This text is about economics.” and “This text is about politics.”. The model then evaluates for both hypotheses if they are entailed in the provided text or not. # noqa: E501
    """

    api_token: SecretStr
    """ API access token for HuggingFace."""

    llm: InferenceClient

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = InferenceClient(
            self.model, api_key=self.api_token.get_secret_value()
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run zero-shot classification on the given input.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut
                off at the first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising
                NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are
                usually passed to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT
            include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        classification_result = self.llm.zero_shot_classification(
            text=prompt,
            candidate_labels=self.labels,
            multi_label=self.multi_label,
            hypothesis_template=self.hypothesis_template,
        )
        print(f"zero shot classification return: " f"{classification_result[0].label}")
        # we simply return the first result element (which has by
        # convention the highest probability (?))
        return classification_result[0].label

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model.
        Used for logging purposes only."""
        return f"Zero Shot Classifier ({self.model})"


def _current_subdir(subdirectory_construction: str | None) -> str:
    if (
        subdirectory_construction is None
        or subdirectory_construction not in SubdirConstruction.value2formatcode().keys()
    ):
        return ""
    now = datetime.now()
    dateformat = SubdirConstruction.value2formatcode()[subdirectory_construction]
    return now.strftime(dateformat)


def _yaml_dump(
    dir_path: str,
    file_name: str,
    dict: Dict,
    local_serialization: bool,
    make_dirs: bool = False,
    key_name: str | None = None,
):
    file_path = os.path.join(dir_path, file_name)
    if local_serialization:
        if make_dirs:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                dict, f, allow_unicode=True, default_flow_style=False, encoding="utf-8"
            )
    else:
        file_path = os.path.join("hf://datasets", dir_path, file_name).replace(
            "\\", "/"
        )
        if key_name:
            fs = HfFileSystem(token=os.environ[key_name])
        else:
            fs = HfFileSystem()
        if make_dirs:
            fs.makedirs(dir_path, exist_ok=True)
        with fs._open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                dict, f, allow_unicode=True, default_flow_style=False, encoding="utf-8"
            )


def _yaml_load(
    file_path: str, local_serialization: bool, key_name: str | None = None
) -> Dict:
    if local_serialization:
        with open(file_path, "r", encoding="utf-8") as f:
            ret_dict = yaml.safe_load(f)
    else:
        file_path = os.path.join("hf://datasets", file_path).replace("\\", "/")
        if key_name:
            fs = HfFileSystem(token=os.environ[key_name])
        else:
            fs = HfFileSystem()
        with fs._open(file_path, "rb", encoding="utf-8") as f:
            ret_dict = yaml.safe_load(f)
    return ret_dict


def _str_load(
    file_path: str, local_serialization: bool, key_name: str | None = None
) -> str:
    logger.info(f"Getting file {file_path}")
    if local_serialization:
        with open(file_path, encoding="utf-8") as f:
            # Read the contents of the file into a variable
            f_str = f.read()
    else:
        file_path = os.path.join("hf://datasets", file_path).replace("\\", "/")
        if key_name:
            fs = HfFileSystem(token=os.environ[key_name])
        else:
            fs = HfFileSystem()
        with fs._open(file_path, "rb", encoding="utf-8") as f:
            # Read the contents of the file into a variable
            f_str = f.read()
    logger.info(f"Got file {file_path}")
    return f_str


def get_toxicity_example_data(
    app_config: AppConfig, data_file: str | None = None
) -> pd.DataFrame:
    if not data_file:
        data_file = app_config.toxicity_examples_data_file
    example_data_file_path = "/".join(
        [
            "hf://datasets",
            app_config.toxicity_examples_hf_base_path,
            app_config.toxicity_examples_data_file,
        ]
    )

    if app_config.toxicity_examples_key_name:
        fs = HfFileSystem(token=os.environ[app_config.toxicity_examples_key_name])
    else:
        fs = HfFileSystem()
    with fs._open(example_data_file_path, "rb") as f:
        example_data_df = pd.read_csv(f)
    logger.info("Loading examples done.")
    return pd.DataFrame(example_data_df[["text", "source"]])
    # return example_data_df['text'].tolist()


def _str_dump(
    file_path: str,
    file_str: str,
    local_serialization: bool,
    key_name: str | None = None,
):
    if local_serialization:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_str)
    else:
        file_path = os.path.join("hf://datasets", file_path).replace("\\", "/")
        if key_name:
            fs = HfFileSystem(token=os.environ[key_name])
        else:
            fs = HfFileSystem()
        with fs._open(file_path, "w", encoding="utf-8") as f:
            f.write(file_str)


def dump_pipeline_config_str(
    file_name: str, pipeline_config_file_str: str, app_config: AppConfig
):
    file_path = os.path.join(
        app_config.get_pipeline_config_path(),
        file_name,
    )
    _str_dump(file_path, pipeline_config_file_str, app_config.local_pipeline_config)


def config_file_exists(app_config: AppConfig, config_file_name: str) -> bool:
    local_serialization = app_config.local_pipeline_config
    if local_serialization:
        config_file_path = os.path.join(
            app_config.get_pipeline_config_path(), config_file_name
        )
        return os.path.isfile(config_file_path)
    else:
        config_file_path = "/".join(
            ["hf://datasets", app_config.get_pipeline_config_path(), config_file_name]
        )
        if app_config.pipeline_config_key_name:
            fs = HfFileSystem(token=os.environ[app_config.pipeline_config_key_name])
        else:
            fs = HfFileSystem()
        return fs.exists(config_file_path)


def pipeline_config_as_string(
    app_config: AppConfig, pipeline_config_file_name: str | None = None
) -> str:
    if not pipeline_config_file_name:
        pipeline_config_file_name = app_config.default_pipeline_config_file
    return _str_load(
        os.path.join(app_config.get_pipeline_config_path(), pipeline_config_file_name),
        app_config.local_pipeline_config,
        app_config.pipeline_config_key_name,
    )


def pipeline_config_file_names(
    app_config: AppConfig, config_version: str | None = None
):
    local_serialization = app_config.local_pipeline_config
    if not config_version:
        config_version = app_config.pipeline_config_version
    config_file_names = []

    config_dir_path = app_config.get_pipeline_config_path()
    if local_serialization:
        for config_file_path in glob(f"{config_dir_path}/*.yaml"):
            with open(config_file_path, "rt", encoding="utf-8") as file:
                file_dict = yaml.safe_load(file)
                if file_dict.get("config_version") == config_version:
                    config_file_names.append(os.path.basename(config_file_path))
    else:
        if app_config.pipeline_config_key_name:
            fs = HfFileSystem(token=os.environ[app_config.pipeline_config_key_name])
        else:
            fs = HfFileSystem()
        config_dir_path = f"hf://datasets/{config_dir_path}".replace("\\", "/")
        for config_file_path in fs.glob(f"{config_dir_path}/*.yaml"):
            with fs._open(config_file_path, "rb", encoding="utf-8") as file:
                file_dict = yaml.safe_load(file)
                if file_dict.get("config_version") == config_version:
                    config_file_names.append(os.path.basename(config_file_path))

    return config_file_names


def update_feedback(
    pipeline_config: PipelineConfig,
    result: ToxicityDetectorResult,
    feedback_text: str | None = None,
    feedback_correctness: str | None = None,
    feedback_likert_content: Dict | None = None,
):

    result.feedback["feedback_text"] = feedback_text
    result.feedback["correctness"] = feedback_correctness
    if feedback_likert_content:
        for key, value in feedback_likert_content.items():
            result.feedback[key] = value

    save_result(result, pipeline_config)


def save_result(result: ToxicityDetectorResult, pipeline_config: PipelineConfig):
    subdirectory_path = _current_subdir(pipeline_config.subdirectory_construction)
    local_serialization = pipeline_config.local_serialization
    file_name = f"{result.request_id}.yaml"
    dir_path = os.path.join(
        pipeline_config.get_base_path(),
        pipeline_config.result_data_path,
        subdirectory_path,
    )
    _yaml_dump(
        dir_path,
        file_name,
        result.model_dump(),
        local_serialization,
        make_dirs=True,
        key_name=pipeline_config.hf_key_name,
    )


def get_openai_chat_model(
    base_url: str,
    model: str,
    api_key: SecretStr,
) -> ChatOpenAI:
    """Return ChatOpenAI model with given server URL, model, API key.

    Args:
    base_url (str): The URL of the inference server.
    model (str): Model (repo id).
    api_key (SecretStr): The API key for accessing the inference
        server.

    Returns:
    model (ChatOpenAI): An instance of the ChatOpenAI model.
    """
    chat_model = ChatOpenAI(base_url=base_url, model=model, api_key=api_key)

    return chat_model
