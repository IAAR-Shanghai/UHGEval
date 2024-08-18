import os
import random

from ..base_evaluator import BaseEvaluator
from ...llms.base_llm import BaseLLM
from .dataset import HaluEvalDataset


class BaseHaluEvalEvaluator(BaseEvaluator):
    """Base class for HaluEval tasks."""

    def __init__(
        self,
        model: BaseLLM,
        num_batches: int = 10,
        output_dir: str = "./output",
        random_seed: int = 0,
    ):
        random.seed(random_seed)
        super().__init__(model, num_batches, output_dir, random_seed=random_seed)

    def set_generation_configs(self):
        new_configs = {
            "temperature": 0.0,
            "max_new_tokens": 8,
        }
        self.model.update_generation_configs(new_configs)

    def load_batched_dataset(self) -> list[list[dict]]:
        dir = os.path.dirname(__file__)
        if "Dialog" in self.__class__.__name__:
            filename = f"dataset_halueval_dialogue.jsonl"
        elif "QA" in self.__class__.__name__:
            filename = f"dataset_halueval_qa.jsonl"
        elif "Summa" in self.__class__.__name__:
            filename = f"dataset_halueval_summarization.jsonl"
        else:
            raise ValueError("Invalid HaluEval task.")
        path = os.path.join(dir, filename)
        dataset = HaluEvalDataset(path)
        batches = dataset.to_batched(self.num_batches)
        return batches

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            "accuracy": sum([result["metrics"]["correct"] for result in results])
            / len(results),
            "num": len(results),
        }
