from pydantic import (
    BaseModel
)
import enum
from typing import Dict, Optional, List


class ToxicityType(enum.Enum):
    GRUP = "hatespeech"
    PERS = "personalized_toxicity"


class ToxicityAnswer(enum.Enum):
    """Enum for toxicity detection results."""
    TRUE = "true"  # Text contains toxic content
    FALSE = "false"  # Text does not contain toxic content
    UNCLEAR = "unclear"  # Insufficient information to determine


class Task(BaseModel):
    name: str
    llm_description: str
    positive_examples: Optional[List[str]] = None
    negative_examples: Optional[List[str]] = None


class Toxicity(BaseModel):
    title: str
    user_description: str
    llm_description: str
    tasks: Dict[str, Dict[str, Task]]
