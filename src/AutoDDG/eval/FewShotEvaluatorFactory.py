from typing import Any, Dict

from .BaseFewShotEvaluator import BaseFewShotEvaluator
from .DAVEFewShotEvaluator import DAVEFewShotEvaluator


class FewShotEvaluatorFactory:
    @staticmethod
    def create(config: Dict[str, Any], job) -> BaseFewShotEvaluator:
        if config.type == "DAVE":
            return DAVEFewShotEvaluator(config, job)
        else:
            raise ValueError(f"Unsupported FewShotEvaluator type: {config.type}")
