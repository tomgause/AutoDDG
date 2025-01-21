import json
import os
from abc import ABC
from typing import Any, Dict

from ..eval.FewShotEvaluatorFactory import FewShotEvaluatorFactory
from ..utils.logging import setup_logger
from ..utils.random import generate_random_name


class Job(ABC):
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self.id = generate_random_name()
        self.logger = setup_logger(self.__class__.__name__, self.id)
        self.output_dir = os.path.join("jobs", self.id)

        os.makedirs(self.output_dir, exist_ok=True)
        self._write_config()

    @property
    def config(self):
        return self._config

    def _write_config(self):
        with open(os.path.join(self.output_dir, "config.json"), "w") as config_file:
            json.dump(self.config.__dict__, config_file, indent=4)

    def run(self):
        evaluator = FewShotEvaluatorFactory.create(config=self.config, job=self)
        evaluator.evaluate()
