import os

from ...llms.base_llm import BaseLLM
from ..base_evaluator import BaseEvaluator
from .dataset import ExampleQADataset

PROMPT_TEMPLATE = """You will be asked a series of questions. For each question, you need to provide a short answer. Do not provide any additional information.

Q: The capital of Germany is?
A: Berlin

Q: What is the largest planet in our solar system?
A: Jupiter

Q: How many continents are there on Earth? (answer in numeral)
A: 7

Q: What is 17+25? (answer in numeral)
A: 42

Q: {question}
A: """


class ExampleQAEvaluator(BaseEvaluator):

    def __init__(
        self, model: BaseLLM, num_batches: int = 1, output_dir: str = "./output"
    ):
        super().__init__(model, num_batches, output_dir)

    def set_generation_configs(self) -> None:
        new_configs = {"max_new_tokens": 16, "do_sample": False}
        self.model.update_generation_configs(new_configs)

    def load_batched_dataset(self) -> list[list[dict]]:
        dir = os.path.dirname(__file__)
        filename = f"dataset_exampleqa.jsonl"
        path = os.path.join(dir, filename)
        dataset = ExampleQADataset(path)
        batches = dataset.to_batched(self.num_batches)
        return batches

    def scoring(self, data_point: dict) -> dict:
        query = PROMPT_TEMPLATE.format(question=data_point["q"])
        response = self.model.safe_request(query)
        answer = response.strip().split("\n")[0].strip()  # Get the first line
        return {
            "metrics": {
                "correct": answer == data_point["a"],
            },
            "log": {
                "answer": answer,
                "response": response,
            },
            "valid": answer != "",
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            "accuracy": sum([result["metrics"]["correct"] for result in results])
            / len(results),
            "num": len(results),
        }
