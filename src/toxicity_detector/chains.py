from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from operator import itemgetter

from langchain_core.runnables import (
    Runnable,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class BaseChainBuilder(ABC):
    """Abstract Base Class for chain builders based on langchain"""

    # TODO: How to handle diverging model kwargs?
    # How to individually configure models?
    @classmethod
    @abstractmethod
    def build(
        cls, llms_dict: Dict[str, BaseLanguageModel], **model_kwargs
    ) -> Runnable:
        """Abstract factory method to build an LLM chain.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): The models to be
                used (names, keys are given by convention).

        Returns:
            Runnable: Chain
        """
        pass


class IdentifyToxicContentZeroShotChain(BaseChainBuilder):
    """Chain using chat model to explain/justify toxicity categorisation."""

    # Chain builder

    # Chat prompts

    _prompt_explain_toxicity = [
        ("system", "{system_prompt}"),
        (
            "user",
            (
                "Task: The text below was characterized as "
                "{toxicity_value}. Suppose that this evaluation is true.\n"
                "Justify this evaluation."
                "Read the following background information carefully "
                "before answering!\n"
                "/// Background_information about toxic content:\n"
                "{toxicity_explication}"
                "\n///\n"
                "/// The text that was characterized as {toxicity_value}:\n"
                "{user_input}\n"
                "\n///\n"
                "Can you please explain why the given characterization "
                "is correct?"
            ),
        ),
    ]

    @classmethod
    def build(
        cls, llms_dict: Dict[str, BaseLanguageModel], **model_kwargs
    ) -> Runnable:
        """Simple chain based on zero-shot model for categorization.

        Uses a chat model for ex post justification of the categorization.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): A dict with one
                model of the form `{'zero_shot_model': the_zero_shot_model,
                'chat_model': the_chat_model}`

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
                    "toxicity_explication": inputs["passed"][
                        "toxicity_explication"
                    ],
                    "user_input": inputs["passed"]["user_input"],
                }
            )
            | ChatPromptTemplate.from_messages(cls._prompt_explain_toxicity)
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

    _prompt_preprocessing = [
        ("system", "{{ system_prompt }}"),
        (
            "user",
            (
                "Aufgabe: Beantworte die folgenden Fragen über den zu "
                "analysierenden Text:\n"
                "{{ general_questions['llm_description'] }}\n"
                "/// Der Text, den Du analysieren sollst:\n"
                "{{ user_input }}\n"
                "\n///\n"
                "{% if context_information %}\n"
                "Beachte für die Analyse die folgenden relevanten "
                "Kontextinformationen:\n"
                "{{ context_information }}.\n"
                "{% endif %}"
                "Hinweise:\n"
                '- Starte die Antwort nicht mit "Ja, ..." bzw. "Nein, ..." '
                "Formuliere die Antworten einfach als Aussagen.\n"
                "- Du musst die Antworten nicht erklären."
            ),
        ),
    ]

    _prompt_indicator_classification = [
        ("system", "{{ system_prompt }}"),
        (
            "user",
            (
                "Aufgabe: Trifft das folgende Merkmal auf den zu "
                "analysierenden Text zu?\n"
                "/// Erläuterung des Merkmals: \n"
                # TODO: Add the name of the indicator here
                "{{ indicator_description }}\n"
                "\n///\n"
                "/// Der Text, den Du analysieren sollst:\n"
                "{{ user_input }}\n"
                "\n///\n"
                "{% if context_information %}\n"
                "Beachte für die Analyse die folgenden relevanten "
                "Kontextinformationen:\n"
                "{{ context_information }}.\n"
                "{% endif %}"
                "Ein vorherige Analyse ergab bereits die folgenden "
                "vorläufigen Analyseergebnisse:\n"
                "{{ preprocessing_results }}\n"
                "Hinweise:\n"
                "- Beachte für die Beantwortung die vorher genannten "
                "vorläufigen Analyseergebnisse!\n"
                '- Starte die Antwort nicht mit "Ja, ..." bzw. "Nein, ..." '
                "Formliere die Antworten einfach als Aussagen.\n"
                "- Formuliere bitte eine kurze Erläuterung bzw. "
                "Begründung für deine Einschätzung."
            ),
        ),
    ]

    _prompt_indicator_aggregation = [
        ("system", "{{ system_prompt }}"),
        (
            "user",
            (
                "Aufgabe: Analysiere ob der untere Text toxischen Inhalt "
                "enthält.\n"
                "Beachte für deine Analyse die folgende "
                "Begriffserläuterung von 'Toxizität':\n"
                "/// Toxizitätserläuterung:\n"
                "{{ toxicity_explication }}"
                "\n///\n"
                "/// Der Text, den du analysieren sollts:\n"
                "{{ user_input }}\n"
                "\n///\n"
                "{% if context_information %}\n"
                "Beachte für die Analyse die folgenden relevanten "
                "Kontextinformationen:\n"
                "{{ context_information }}.\n"
                "{% endif %}"
                "Beachte für deine Analyse die folgendenden vorläufigen "
                "Analyseergebnisse:\n"
                "{{ preprocessing_results }}\n"
                "Eine Indikatorenanalyse ergab:\n"
                "{% for indicator_key, indicator_result in "
                "indicator_analysis.items() %}"
                "- {{ indicators_dict[indicator_key]['name']}}: "
                "{{ indicator_result }}\n"
                "{% endfor %}"
                "Analysiere nun bitte den Text und begründe deine "
                "Einschätzung!"
            ),
        ),
    ]
    _formatting_prompt_msgs = [
        ("system", "{system_prompt}"),
        ("user", "Analysiere ob der untere Text toxischen Inhalt enthält."),
        ("assistant", "{analysis_result}"),
        (
            "user",
            "Danke! Fasse die Antwort nun als Kurzantwort in einem Wort "
            "zusammen:\n"
            '"True", wenn der Text toxischen Inhalt enthält,\n'
            '"False", wenn der Text keinen toxischen Inhalt enthält,\n'
            "Just return the word, without the quotation marks.",
        ),
    ]

    @classmethod
    def prompts(cls, **kwargs: Any) -> Dict:
        ret_dict = {
            "preprocessing": ChatPromptTemplate.from_messages(
                cls._prompt_preprocessing, template_format="jinja2"
            )
            .format_prompt(**kwargs)
            .to_string(),
            "indicators": {
                indicator_key: ChatPromptTemplate.from_messages(
                    cls._prompt_indicator_classification,
                    template_format="jinja2",
                )
                .format_prompt(
                    indicator_description=indicator["llm_description"],
                    indicator_name=indicator["name"],
                    **kwargs,
                )
                .to_string()
                for indicator_key, indicator in kwargs[
                    "indicators_dict"
                ].items()
            },
            "aggregation": ChatPromptTemplate.from_messages(
                cls._prompt_indicator_aggregation, template_format="jinja2"
            )
            .format_prompt(**kwargs)
            .to_string(),
        }
        return ret_dict

    @classmethod
    def build(
        cls,
        llms_dict: Dict[str, BaseLanguageModel],
        indicators_dict: Dict[str, Dict[str, str]] | None = None,
        **model_kwargs,
    ) -> Runnable:
        """Builds a chain for identifying toxicity.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): A dict with one
                model of the form `{'chat_model': the_chat_model}`

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
            ChatPromptTemplate.from_messages(
                cls._prompt_preprocessing, template_format="jinja2"
            )
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
                    cls._prompt_indicator_classification,
                    template_format="jinja2",
                )
                | llm.bind(**model_kwargs)
                | StrOutputParser()
            )
            return chain

        # sub chain: aggregation chain
        aggregation_chain = (
            ChatPromptTemplate.from_messages(
                cls._prompt_indicator_aggregation, template_format="jinja2"
            )
            | llm.bind(**model_kwargs)
            | StrOutputParser()
        )

        main_chain = (
            # We add the indicators dict to the
            # General questions chain (preprocessing)
            RunnablePassthrough.assign(
                preprocessing_results=general_props_chain
            )
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
                    cls._formatting_prompt_msgs
                )
                # TODO: Perhaps, set temperature to 0
                | llm.bind(**model_kwargs)
                | StrOutputParser()
                | (lambda input: True if input.lower() in ["true"] else False)
            )
        )

        return main_chain


class IdentifyToxicContentChatChain(BaseChainBuilder):
    """Chain that uses a Chatmodel for the categorisation."""

    # Chat prompts

    _prompt_identify_toxicity = [
        ("system", "{system_prompt}"),
        (
            "user",
            (
                "Task: Identify toxic content in the text below.\n"
                "Read the following background information carefully "
                "before answering!\n"
                "/// Background_information about toxic content:\n"
                "{toxicity_explication}"
                "\n///\n"
                "/// The text that you should analyse:\n"
                "{user_input}\n"
                "\n///\n"
                "Can you please identify toxic content and explain your "
                "decisions?"
            ),
        ),
    ]
    # Preprocessing methods

    pass

    # Postprocessing methods

    pass

    # for debuggin/loggin only:
    @classmethod
    def format_prompt(cls, **kwargs: Any) -> str:
        return (
            ChatPromptTemplate.from_messages(cls._prompt_identify_toxicity)
            .format_prompt(**kwargs)
            .to_string()
        )

    # Chain builder

    @classmethod
    def build(
        cls, llms_dict: Dict[str, BaseLanguageModel], **model_kwargs
    ) -> Runnable:
        """Builds a simplistic chain for identifying toxicity.

        Args:
            llms_dict (Dict[str, BaseLanguageModel]): A dict with one
                model of the form `{'chat_model': the_chat_model}`

        Returns:
            Runnable: Chain
        """
        main_chain = (
            ChatPromptTemplate.from_messages(cls._prompt_identify_toxicity)
            | llms_dict["chat_model"].bind(**model_kwargs)
            | StrOutputParser()
        )

        return main_chain
