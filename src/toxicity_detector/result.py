import pydantic
import uuid
from typing import Any, Dict
from datetime import datetime, timezone

from toxicity_detector.config import PipelineConfig


class ToxicityDetectorResult(pydantic.BaseModel):
    user_input: str | None = None
    user_input_source: str | None = None
    context_information: str | None = None
    toxicity_type: str | None = None
    answer: Dict[str, Any] = {}
    pipeline_config: PipelineConfig | None = None
    feedback: Dict[str, Any] = {}
    request_id: str = pydantic.Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    request_time: str = pydantic.Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC"))
