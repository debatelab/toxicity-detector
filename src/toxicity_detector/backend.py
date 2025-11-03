from glob import glob
import os
from langchain_huggingface import HuggingFaceEndpoint
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from huggingface_hub import InferenceClient
import yaml
import uuid
from huggingface_hub import HfFileSystem
import datetime
import pandas as pd

from toxicity_detector import MonoModelDetectToxicityChain


def detect_toxicity(
    input_text: str,
    user_input_source: str | None,
    toxicity_type: str,
    context_info: str | None,
    model_config_dict: Dict,
    app_config_dict: Dict,
    serialize_result: bool = True,
):

    def log_msg(msg: str):
        log_message(msg, app_config_dict)

    if not input_text or input_text == "":
        raise ValueError("Input text must not be empty.")

    # create new uuid for the detection request
    # (used for UI logic to attach user feedback and for data serialization)
    detection_id = get_request_uuid(app_config_dict)
    log_msg(f"Starting new detection request (uuid: {detection_id}).")
    context_info = (
        None if not context_info or context_info.isspace() else context_info
    )
    model = model_config_dict["used_chat_model"]

    log_msg(f"Chosen toxicity type: {toxicity_type}")
    log_msg(f"Used model: {model_config_dict['models'][model]['name']}")
    log_msg(f"Kontextinfo: {context_info}")
    # Chat model
    if model_config_dict["models"][model]["llm_chain"] == "chat-chain":
        # getting api key
        if "api_key" in model_config_dict["models"][model].keys():
            api_key = SecretStr(model_config_dict["models"][model]["api_key"])
        elif "api_key_name" in model_config_dict["models"][model].keys():
            api_key_name = model_config_dict["models"][model]["api_key_name"]
            log_msg(f"Used api key name: {api_key_name}")
            # check whether the api key is set as env variable
            if os.environ.get(api_key_name) is None:
                raise ValueError(
                    f"The api key name {api_key_name} is not set as "
                    f"env variable."
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
        if "model_kwargs" in model_config_dict["models"][model].keys():
            model_kwargs = model_config_dict["models"][model]["model_kwargs"]

        log_msg(
            f"Model kwargs: {model_kwargs}"
        )
        # building chain
        toxicitiy_detection_chain = MonoModelDetectToxicityChain.build(
            llms_dict={
                "chat_model": get_openai_chat_model(
                    api_key=api_key,
                    model=model_config_dict["models"][model]["model"],
                    base_url=model_config_dict["models"][model]["base_url"],
                )
            },
            indicators_dict=model_config_dict["toxicities"][toxicity_type][
                "tasks"
            ]["indicator_analysis"],
            **model_kwargs
        )
    else:
        # TODO: log warning
        raise ValueError(
            f"llm_chain "
            f"{model_config_dict['models'][model]['llm_chain']} not "
            f"implemented."
        )
    # TODO: Offering possibility to use zero shot classifiers
    # (for, e.g., indicator analysis)
    # using zero-shot classifier
    # elif model_config_dict['models'][model]['llm_chain'] == \
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
    #             model=model_config_dict['models'][model]['repo_id'],
    #             labels=list(model_config_dict['toxicities'][
    #                 toxicity_type]['zero-shot-categorization'][
    #                 'labels'].values()),
    #             multi_label=model_config_dict['toxicities'][toxicity_type]['zero-shot-categorization']['multi_label'],
    #             hypothesis_template=model_config_dict['toxicities'][toxicity_type]['zero-shot-categorization']['hypothesis_template'],
    #             api_token=SecretStr(os.environ.get("HF_TOKEN_KIDEKU_INFERENCE"))
    #         ),
    #         'chat_model': backend.get_chat_model(
    #             token=os.environ.get("HF_TOKEN_KIDEKU_INFERENCE"),
    #             repo_id=model_config_dict['models'][explaining_model_name]['repo_id']
    #         )
    #     })

    answer = toxicitiy_detection_chain.invoke(
        {
            "system_prompt": model_config_dict["system_prompt"],
            "toxicity_explication": model_config_dict["toxicities"][
                toxicity_type
            ]["llm_description"],
            "user_input": input_text,
            "user_input_source": user_input_source,
            "general_questions": model_config_dict["toxicities"][
                toxicity_type
            ]["tasks"]["prepatory_analysis"]["general_questions"],
            "context_information": context_info,
            "indicators_dict": model_config_dict["toxicities"][toxicity_type][
                "tasks"
            ]["indicator_analysis"],
        }
    )

    answer["toxicity_type"] = toxicity_type
    answer["date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    answer["uid"] = detection_id
    result_dict = {"config": model_config_dict, "query": answer}
    # saving the result
    # TODO: as async
    if serialize_result:
        save_result(
            result_dict=result_dict,
            file_name=f"{detection_id}.yaml",
            app_config_dict=app_config_dict,
        )

    return result_dict


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
            self.model,
            api_key=self.api_token.get_secret_value()
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
            labels=self.labels,
            multi_label=self.multi_label,
            hypothesis_template=self.hypothesis_template,
        )
        print(
            f"zero shot classification return: "
            f"{classification_result[0].label}"
        )
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


def log_message(message: str, app_config_dict: Dict):
    local_serialization = app_config_dict["data_serialization"][
        "local_serialization"
    ]
    log_dir_path = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["log_path"],
    )

    now = datetime.datetime.now()
    log_file_name = now.strftime("toxicity_detector_log_%Y_%m_%d.log")
    log_file_path = os.path.join(log_dir_path, log_file_name)

    if local_serialization:
        log_dir_path = os.path.join(
            app_config_dict["data_serialization"]["local_base_path"],
            app_config_dict["data_serialization"]["log_path"],
        )
        os.makedirs(log_dir_path, exist_ok=True)
        with open(log_file_path, "a") as log_file:
            log_file.write(message + "\n")
    print(message)


def _current_subdir(app_config_dict: Dict) -> str:
    """Generate subdir path based on date and construction method.

    Generates a subdirectory path based on the current date
    and the specified subdirectory construction method.

    Args:
        app_config_dict (Dict): A dictionary containing application
            configuration. It must have a key 'data_serialization'
            which itself contains a key 'subdirectory_construction'.
            The value of 'subdirectory_construction' should be one of
            None, "monthly", "weekly", "daily", or "yearly".

    Returns:
        str: A string representing the subdirectory path based on
            the current date and the specified subdirectory
            construction method. If 'subdirectory_construction' is
            None, an empty string is returned.

    Raises:
        ValueError: If 'subdirectory_construction' is not one of
            None, "monthly", "weekly", "daily", or "yearly".
    """
    subdirectory_construction = app_config_dict["data_serialization"][
        "subdirectory_construction"
    ]

    if subdirectory_construction not in [
        None, "monthly", "weekly", "yearly", "daily"
    ]:
        raise ValueError(
            "subdirectory must be one of None, 'monthly', "
            "'weekly', 'daily', or 'yearly'."
        )

    now = datetime.datetime.now()
    if subdirectory_construction == "monthly":
        subdirectory_path = now.strftime("y%Y_m%m")
    elif subdirectory_construction == "weekly":
        year, week, _ = now.isocalendar()
        subdirectory_path = f"y{year}_w{week}"
    elif subdirectory_construction == "yearly":
        subdirectory_path = now.strftime("y%Y")
    elif subdirectory_construction == "daily":
        subdirectory_path = now.strftime("%Y_%m_%d")
    else:
        subdirectory_path = ""
    return subdirectory_path


def _yaml_dump(
    dir_path: str,
    file_name: str,
    dict: Dict,
    local_serialization: bool,
    make_dirs: bool = False,
):
    file_path = os.path.join(dir_path, file_name)
    if local_serialization:
        if make_dirs:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                dict, f, allow_unicode=True,
                default_flow_style=False, encoding="utf-8"
            )
    else:
        fs = HfFileSystem()
        if make_dirs:
            fs.makedirs(dir_path, exist_ok=True)
        with fs.open(file_path, "w", encoding="UTF8") as f:
            yaml.dump(
                dict, f, allow_unicode=True,
                default_flow_style=False, encoding="utf-8"
            )


def _yaml_load(file_path: str, local_serialization: bool) -> Dict:
    if local_serialization:
        with open(file_path, "r") as f:
            ret_dict = yaml.safe_load(f)
    else:
        fs = HfFileSystem()
        with fs.open(file_path, "r") as f:
            ret_dict = yaml.safe_load(f)
    return ret_dict


def _str_load(file_path: str, local_serialization: bool) -> str:
    if local_serialization:
        with open(file_path) as f:
            # Read the contents of the file into a variable
            f_str = f.read()
    else:
        fs = HfFileSystem()
        with fs.open(file_path, "rt") as f:
            # Read the contents of the file into a variable
            f_str = f.read()
    return f_str


def get_toxicity_example_data(
    app_config_dict: Dict, data_file: str = None
) -> List[str]:
    if not data_file:
        data_file = app_config_dict["data_serialization"][
            "toxicity_examples_data_file"
        ]
    example_data_file_path = os.path.join(
        app_config_dict["data_serialization"]["hf_base_path"],
        app_config_dict["data_serialization"]["toxicity_examples_data_path"],
        data_file,
    )

    fs = HfFileSystem()
    with fs.open(example_data_file_path, "r") as f:
        example_data_df = pd.read_csv(f)
    return example_data_df
    # return example_data_df['text'].tolist()


def _str_dump(file_path: str, file_str: str, local_serialization: bool):
    if local_serialization:
        with open(file_path, "w") as f:
            f.write(file_str)
    else:
        fs = HfFileSystem()
        with fs.open(file_path, "w") as f:
            f.write(file_str)


def dump_model_config_str(
    file_name: str, model_config_file_str: str, app_config_dict: Dict
):
    data_ser = app_config_dict["data_serialization"]
    local_serialization = data_ser["local_serialization"]
    file_path = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["model_config_path"],
        file_name,
    )
    _str_dump(file_path, model_config_file_str, local_serialization)


def config_file_exists(app_config_dict: Dict, config_file_name: str) -> bool:
    data_ser = app_config_dict["data_serialization"]
    local_serialization = data_ser["local_serialization"]
    config_dir_path = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["model_config_path"],
    )
    config_file_path = os.path.join(config_dir_path, config_file_name)

    if local_serialization:
        return os.path.isfile(config_file_path)
    else:
        fs = HfFileSystem()
        return fs.exists(config_file_path)


def model_config_as_string(
    app_config_dict: Dict, model_config_file_name: str | None = None
) -> str:
    data_ser = app_config_dict["data_serialization"]
    local_serialization = data_ser["local_serialization"]
    if not model_config_file_name:
        model_config_file_name = app_config_dict["default_model_config_file"]
    config_dir_path = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["model_config_path"],
    )
    return _str_load(
        os.path.join(config_dir_path, model_config_file_name),
        local_serialization
    )


def model_config(
    app_config_dict: Dict,
    model_config_file_name: str | None = None,
) -> Dict:
    return yaml.safe_load(
        model_config_as_string(app_config_dict, model_config_file_name)
    )


def model_config_file_names(app_config_dict: Dict, config_version: str | None = None):
    data_ser = app_config_dict["data_serialization"]
    local_serialization = data_ser["local_serialization"]
    if not config_version:
        config_version = app_config_dict["model_config_version"]
    config_file_names = []
    config_dir_path = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["model_config_path"],
    )
    if local_serialization:
        for config_file_path in glob(f"{config_dir_path}/*.yaml"):
            with open(config_file_path, "rt") as file:
                file_dict = yaml.safe_load(file)
                if file_dict.get("config_version") == config_version:
                    config_file_names.append(
                        os.path.basename(config_file_path)
                    )
    else:
        fs = HfFileSystem()
        for config_file_path in fs.glob(f"{config_dir_path}/*.yaml"):
            with fs.open(config_file_path, "rt") as file:
                file_dict = yaml.safe_load(file)
                if file_dict.get("config_version") == config_version:
                    config_file_names.append(
                        os.path.basename(config_file_path)
                    )

    return config_file_names


def update_feedback(
    app_config_dict: Dict,
    detection_id: str,
    feedback_text: str | None = None,
    feedback_correctness: str | None = None,
    feedback_likert_content: Dict | None = None,
):
    # TODO: better find file by detection_id
    subdirectory_path = _current_subdir(app_config_dict)
    data_ser = app_config_dict["data_serialization"]
    local_serialization = data_ser["local_serialization"]
    dir_path = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["result_data_path"],
        subdirectory_path,
    )

    file_name = f"{detection_id}.yaml"
    result_file_path = os.path.join(dir_path, f"{detection_id}.yaml")
    result_dict = _yaml_load(result_file_path, local_serialization)

    if "feedback" not in result_dict.keys():
        result_dict["feedback"] = {}

    result_dict["feedback"]["feedback_text"] = feedback_text
    result_dict["feedback"]["correctness"] = feedback_correctness
    if feedback_likert_content:
        for key, value in feedback_likert_content.items():
            result_dict["feedback"][key] = value

    _yaml_dump(dir_path, file_name, result_dict, local_serialization)


def save_result(result_dict: Dict, file_name: str, app_config_dict: Dict):
    # adding prompts for logging
    prompts = MonoModelDetectToxicityChain.prompts(
        **result_dict["query"]
    )
    result_dict["query"]["prompts"] = prompts

    subdirectory_path = _current_subdir(app_config_dict)
    data_ser = app_config_dict["data_serialization"]
    local_serialization = data_ser["local_serialization"]

    dir_path = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["result_data_path"],
        subdirectory_path,
    )
    _yaml_dump(
        dir_path, file_name, result_dict, local_serialization,
        make_dirs=True
    )

# def get_chat_model(token: str, repo_id: str):
#     llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=token)
#     # chat_model = ChatHuggingFace(llm=llm)
#     return llm

def get_request_uuid(app_config_dict: Dict) -> str:
    # Finding existing uuids.
    data_ser = app_config_dict["data_serialization"]
    local_serialization = data_ser["local_serialization"]
    data_dir = os.path.join(
        (
            app_config_dict["data_serialization"]["local_base_path"]
            if local_serialization
            else app_config_dict["data_serialization"]["hf_base_path"]
        ),
        app_config_dict["data_serialization"]["result_data_path"],
    )
    # On local file system
    if app_config_dict["data_serialization"]["local_serialization"]:
        uiids = [
            os.path.splitext(os.path.basename(file))[0]
            for file in glob(f"{data_dir}/**/*.yaml", recursive=True)
        ]
    # On HF
    else:
        fs = HfFileSystem()
        if not fs.exists(data_dir):
            return str(uuid.uuid4())
        uiids = [
            os.path.splitext(os.path.basename(file))[0]
            for file in fs.glob(f"{data_dir}/**/*.yaml")
        ]
    # print(f"Existing uuids: {uiids}")
    # TODO: Uniqueness is already guaranteed by the function itself
    unique_id = str(uuid.uuid4())
    while unique_id in uiids:
        unique_id = str(uuid.uuid4())
    return unique_id


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
