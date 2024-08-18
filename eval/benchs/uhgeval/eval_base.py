import os

from ..base_evaluator import BaseEvaluator
from ...llms.base_llm import BaseLLM
from .dataset import UHGEvalDataset


class BaseUHGEvaluator(BaseEvaluator):
    """Base class for UHGEval tasks.

    Note:
        `use_full` is a boolean flag that determines whether to use the full
        dataset or a concise version of it.
    """

    def __init__(
        self,
        model: BaseLLM,
        num_batches: int = 10,
        output_dir: str = "./output",
        use_full: bool = False,
    ):
        super().__init__(model, num_batches, output_dir, use_full=use_full)

    def set_generation_configs(self):
        new_configs = {
            "temperature": 0.1,
            "max_new_tokens": 24,
            "top_p": 0.9,
            "top_k": 5,
        }
        self.model.update_generation_configs(new_configs)

    def load_batched_dataset(self) -> list[list[dict]]:
        dir = os.path.dirname(__file__)
        mode = "full" if self.more["use_full"] else "concise"
        filename = f"dataset_uhgeval_{mode}.jsonl"
        path = os.path.join(dir, filename)
        dataset = UHGEvalDataset(path)
        batches = dataset.to_batched(self.num_batches)
        return batches
