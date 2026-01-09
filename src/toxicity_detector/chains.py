from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import re

from operator import itemgetter

from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from toxicity_detector.datamodels import ToxicityAnswer


class BaseChainBuilder(ABC):
    """Abstract Base Class for chain builders based on langchain"""

    # TODO: How to handle diverging model kwargs?
    # How to individually configure models?
    @classmethod
    @abstractmethod
    def build(cls, llms_dict: Dict[str, BaseLanguageModel], **kwargs) -> Runnable:
        """Abstract factory method to build an LLM chain.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): The models to be
                used (names, keys are given by convention).
            **kwargs: Additional parameters including prompts and model kwargs.

        Returns:
            Runnable: Chain
        """
        pass


class IdentifyToxicContentZeroShotChain(BaseChainBuilder):
    """Chain using chat model to explain/justify toxicity categorisation."""

    @classmethod
    def build(  # type: ignore[override]
        cls,
        llms_dict: Dict[str, BaseLanguageModel],
        explain_toxicity: list[tuple[str, str]],
        **model_kwargs,
    ) -> Runnable:
        """Simple chain based on zero-shot model for categorization.

        Uses a chat model for ex post justification of the categorization.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): A dict with one
                model of the form `{'zero_shot_model': the_zero_shot_model,
                'chat_model': the_chat_model}`
            explain_toxicity: List of (role, content) tuples for the prompt

        Returns:
            Runnable: Chain
        """
        main_chain = (
            RunnableParallel(
                passed=RunnablePassthrough(),
                zero_shot_classification=itemgetter("user_input")
                | llms_dict["zero_shot_model"].bind(**model_kwargs),
            )
            | (
                lambda inputs: {
                    "toxicity_value": (
                        "toxic"
                        if inputs["zero_shot_classification"]
                        == inputs["passed"]["labels"]["toxic"]
                        else "not toxic"
                    ),
                    "toxicity_explication": inputs["passed"]["toxicity_explication"],
                    "user_input": inputs["passed"]["user_input"],
                }
            )
            | ChatPromptTemplate.from_messages(explain_toxicity)
            | llms_dict["chat_model"].bind(**model_kwargs)
        )
        return main_chain


class MonoModelDetectToxicityChain(BaseChainBuilder):
    """Chain builder for detecting toxicity.

    Caveats:
    - This generated chain uses one model only. It is not possible to
      assign different models to the subtasks.

    Input:
    - toxicity_explication: An explication of the toxicity concept.
    - user_input: The text that should be analyzed.
    - general_questions: `Dict[str, str]` of the form
      `{'name': <name>, 'llm_description': <partial prompt>}`.
    - context_information: Context information about the text (may be
      empty).
    """

    @staticmethod
    def _parse_toxicity_response(response: str) -> Optional[ToxicityAnswer]:
        """Parse the toxicity response from the LLM.

        Handles variations in the response format including prefixes like
        "Answer:", "Antwort:", "The answer is:", etc.

        Args:
            response (str): The LLM response to parse

        Returns:
            Optional[ToxicityAnswer]:
                - ToxicityAnswer.TRUE if the text contains toxic content
                - ToxicityAnswer.FALSE if the text doesn't contain toxic content
                - ToxicityAnswer.UNCLEAR if the result is unclear
                - None if the response cannot be parsed
        """
        # Normalize the response: strip whitespace, remove quotes, lowercase
        normalized = response.strip().strip("\"'").lower()

        # Remove common prefixes (English and German)
        prefix_patterns = [
            r"^(answer|antwort|the answer is|die antwort ist|result|ergebnis)\s*:?\s*",
        ]
        for pattern in prefix_patterns:
            normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
        normalized = normalized.strip().strip("\"'")

        # Check for TRUE indicators (toxic content)
        true_indicators = ["true", "yes", "ja", "toxic", "toxisch", "wahr"]
        if normalized in true_indicators:
            return ToxicityAnswer.TRUE

        # Check for FALSE indicators (non-toxic content)
        false_indicators = [
            "false",
            "no",
            "nein",
            "not toxic",
            "non-toxic",
            "nicht toxisch",
            "falsch",
        ]
        if normalized in false_indicators:
            return ToxicityAnswer.FALSE

        # Check for UNCLEAR indicators
        unclear_indicators = [
            "unclear",
            "unklar",
            "unsure",
            "unknown",
            "unbekannt",
            "uncertain",
            "unsicher",
        ]
        if normalized in unclear_indicators:
            return ToxicityAnswer.UNCLEAR

        # If response contains these keywords, extract the meaning
        if any(keyword in normalized for keyword in ["true", "toxic"]):
            # Make sure it's not negated
            if not any(neg in normalized for neg in ["not", "non", "nicht", "kein"]):
                return ToxicityAnswer.TRUE

        if any(
            keyword in normalized
            for keyword in ["false", "not toxic", "non-toxic", "nicht toxisch"]
        ):
            return ToxicityAnswer.FALSE

        if any(keyword in normalized for keyword in ["unclear", "unklar", "unsure"]):
            return ToxicityAnswer.UNCLEAR

        # Default to None if we can't parse the response
        return None

    @classmethod
    def prompts(
        cls,
        preprocessing: list[tuple[str, str]],
        indicator_classification: list[tuple[str, str]],
        indicator_aggregation: list[tuple[str, str]],
        **kwargs: Any,
    ) -> Dict:
        """Generate formatted prompts for debugging/logging.

        Args:
            preprocessing: Preprocessing prompt template
            indicator_classification: Indicator classification
                prompt template
            indicator_aggregation: Aggregation prompt template
            **kwargs: Variables to format into the prompts

        Returns:
            Dict with formatted prompts
        """
        ret_dict = {
            "preprocessing": ChatPromptTemplate.from_messages(
                preprocessing, template_format="jinja2"
            )
            .format_prompt(**kwargs)
            .to_string(),
            "indicators": {
                indicator_key: ChatPromptTemplate.from_messages(
                    indicator_classification,
                    template_format="jinja2",
                )
                .format_prompt(
                    indicator_description=indicator["llm_description"],
                    indicator_name=indicator["name"],
                    **kwargs,
                )
                .to_string()
                for indicator_key, indicator in kwargs["indicators_dict"].items()
            },
            "aggregation": ChatPromptTemplate.from_messages(
                indicator_aggregation, template_format="jinja2"
            )
            .format_prompt(**kwargs)
            .to_string(),
        }
        return ret_dict

    @classmethod
    def build(  # type: ignore[override]
        cls,
        llms_dict: Dict[str, BaseLanguageModel],
        preprocessing: list[tuple[str, str]],
        indicator_classification: list[tuple[str, str]],
        indicator_aggregation: list[tuple[str, str]],
        formatting_prompt_msgs: list[tuple[str, str]],
        indicators_dict: Dict[str, Dict[str, str]] | None = None,
        **model_kwargs,
    ) -> Runnable:
        """Builds a chain for identifying toxicity.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): A dict with one
                model of the form `{'chat_model': the_chat_model}`
            preprocessing: List of (role, content) tuples for
                preprocessing prompt
            indicator_classification: List of (role, content) tuples
                for indicator prompt
            indicator_aggregation: List of (role, content) tuples for
                aggregation prompt
            formatting_prompt_msgs: List of (role, content) tuples for
                formatting prompt
            indicators_dict (Dict[str, Dict[str, str]]): A dict with all
                relevant indicators of the form
                `{<indicator_key>: {'name': <indicator name>,
                'llm_description': <partial prompt>},...}`

        Returns:
            Runnable: Chain
        """
        if indicators_dict is None:
            indicators_dict = {}
        llm = llms_dict["chat_model"]

        # sub chain: preprocessing: analysing text w.r.t.
        # general (relevant) properties
        general_props_chain = (
            ChatPromptTemplate.from_messages(preprocessing, template_format="jinja2")
            | llm.bind(**model_kwargs)
            | StrOutputParser()
        )

        # sub chain: indicator chain
        def indicator_chain(indicator_name: str, indicator_description: str):
            chain = (
                RunnablePassthrough.assign(
                    indicator_name=lambda x: indicator_name,
                    indicator_description=lambda x: indicator_description,
                )
                | ChatPromptTemplate.from_messages(
                    indicator_classification,
                    template_format="jinja2",
                )
                | llm.bind(**model_kwargs)
                | StrOutputParser()
            )
            return chain

        # sub chain: aggregation chain
        aggregation_chain = (
            ChatPromptTemplate.from_messages(
                indicator_aggregation, template_format="jinja2"
            )
            | llm.bind(**model_kwargs)
            | StrOutputParser()
        )

        main_chain = (
            # We add the indicators dict to the
            # General questions chain (preprocessing)
            RunnablePassthrough.assign(preprocessing_results=general_props_chain)
            # Branches: independent indicator chains
            | RunnablePassthrough.assign(
                indicator_analysis=RunnableParallel(
                    {
                        indicator_key: indicator_chain(
                            indicator["name"], indicator["llm_description"]
                        )
                        for indicator_key, indicator in indicators_dict.items()
                    }
                )
            )
            # Aggregation of preliminary results to an overall categorisation
            | RunnablePassthrough.assign(analysis_result=aggregation_chain)
            # Adding binary answer
            | RunnablePassthrough.assign(
                contains_toxicity=ChatPromptTemplate.from_messages(
                    formatting_prompt_msgs, template_format="jinja2"
                )
                # TODO: Perhaps, set temperature to 0
                | llm.bind(**model_kwargs)
                | StrOutputParser()
                | cls._parse_toxicity_response
            )
        )

        return main_chain


class IdentifyToxicContentChatChain(BaseChainBuilder):
    """Chain that uses a Chatmodel for the categorisation."""

    # for debuggin/loggin only:
    @classmethod
    def format_prompt(
        cls, identify_toxicity: list[tuple[str, str]], **kwargs: Any
    ) -> str:
        """Format prompt for debugging/logging.

        Args:
            identify_toxicity: List of (role, content) tuples for the prompt
            **kwargs: Variables to format into the prompt

        Returns:
            Formatted prompt as string
        """
        return (
            ChatPromptTemplate.from_messages(identify_toxicity)
            .format_prompt(**kwargs)
            .to_string()
        )

    @classmethod
    def build(  # type: ignore[override]
        cls,
        llms_dict: Dict[str, BaseLanguageModel],
        identify_toxicity: list[tuple[str, str]],
        **model_kwargs,
    ) -> Runnable:
        """Builds a simplistic chain for identifying toxicity.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): A dict with one
                model of the form `{'chat_model': the_chat_model}`
            identify_toxicity: List of (role, content) tuples for the prompt

        Returns:
            Runnable: Chain
        """
        main_chain = (
            ChatPromptTemplate.from_messages(identify_toxicity)
            | llms_dict["chat_model"].bind(**model_kwargs)
            | StrOutputParser()
        )

        return main_chain
