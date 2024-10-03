import os

from ..base_evaluator import BaseEvaluator
from .dataset import HalluQADataset


class HalluQABaseEvaluator(BaseEvaluator):
    """Base class for HalluQA tasks."""

    def set_generation_configs(self) -> None:
        new_configs = {"max_new_tokens": 4, "do_sample": False}
        self.model.update_generation_configs(new_configs)

    def load_batched_dataset(self) -> list[list[dict]]:
        dir = os.path.dirname(__file__)
        if self.__class__.__name__ == "HalluQAMCEvaluator":
            filename = f"dataset_halluqa_mc.json"
        else:
            filename = f"dataset_halluqa.json"
        path = os.path.join(dir, filename)
        dataset = HalluQADataset(path)
        batches = dataset.to_batched(self.num_batches)
        return batches
