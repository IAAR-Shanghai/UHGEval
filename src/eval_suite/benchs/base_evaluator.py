import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from ..llms.base_llm import BaseLLM
from ..logging import logger


class BaseEvaluator(ABC):
    """Base class for evaluation.

    The class includes four main components:
    1. Initialization: Set up the evaluation environment.
    2. Evaluation Modules: Define the evaluation process.
    3. Engine: Run the evaluation process.
    4. Utilities: Helper functions for evaluation.

    Subclasses must implement the following methods:
    - `set_generation_configs`
    - `load_batched_dataset`
    - `scoring`
    - `compute_overall`
    """

    # ─── Initialization ───────────────────────────────────────────────────

    def __init__(
        self,
        model: BaseLLM,
        num_batches: int = 10,
        output_dir: str = "./output",
        **more,
    ):
        """Copy the parameters only.

        Args:
            model (BaseLLM): The large language model to be evaluated.
            num_batches (int): The dataset will be loaded in batches.
            output_dir (str): The directory for result output and caching.
            more (dict): Additional parameters for the evaluation.
        """
        self.model = model
        self.num_batches = num_batches
        self.output_dir = output_dir
        self.more = more

    @abstractmethod
    def set_generation_configs(self):
        """Use unified parameters for the evaluation."""
        new_configs = {
            # Subclasses should fill in the parameters here.
            # For example,
            # "temperature": 0.1,
            # "max_new_tokens": 8,
            # "top_p": 0.9,
            # "top_k": 5,
        }
        self.model.update_generation_configs(new_configs)

    def get_output_path(self):
        os.makedirs(self.output_dir, exist_ok=True)
        file_name = f"{self.model.model_name}_{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

        invalid_chars_pattern = r'[<>:"/\\|?*\x00-\x1F]'
        sanitized_filename = re.sub(invalid_chars_pattern, "_", file_name)
        sanitized_filename = sanitized_filename.strip()

        output_path = os.path.join(self.output_dir, sanitized_filename)
        return output_path

    @abstractmethod
    def load_batched_dataset(self) -> list[list[dict]]:
        """Load the dataset in batches.

        Returns:
            list[list[dict]]: Batched dataset. Each batch is a list of data points.
        """
        return []

    # ─── Evaluation Modules ───────────────────────────────────────────────

    @abstractmethod
    def scoring(self, data_point: dict) -> dict:
        """Invoke self.model.safe_request to evaluate a data_point.

        Returns:
            dict: A dictionary with the following fields:
            - metrics: Numerical results to be recorded by subclasses, mandatory.
            - log: Any form of results to be recorded by subclasses, optional.
            - valid: Boolean result to be recorded by subclasses, mandatory.
        """
        return {
            "metrics": {
                # Numerical results to be recorded by subclasses, mandatory.
                # Such as accuracy, recall, bleu, rouge, etc.
            },
            "log": {
                # Any form of results to be recorded by subclasses, optional.
                # Such as model output.
            },
            "valid": ...,
            # Boolean result to be recorded by subclasses, indicating whether the evaluation is valid, mandatory.
            # True or False.
        }

    def batch_scoring(self, data_items: list[dict]) -> list[dict]:
        """Perform batch scoring on the given data items."""
        results = []
        for data_point in tqdm(data_items, desc=self.model.model_name):
            results.append(
                {
                    "id": data_point["id"],
                    **self.scoring(data_point),
                    "datetime": datetime.now().isoformat(),
                    "data_point": data_point,
                }
            )
        return results

    @abstractmethod
    def compute_overall(self, results: list[dict]) -> dict:
        """Extract and aggregate results from individual data points in the results.
        For example, calculate mean, variance, etc.

        Args:
            results (list[dict]): A list of results from individual data points.
            The length of the list is ensured to be **greater than 0**.

        Returns:
            dict: A result dictionary that can store any number and form of fields.

        Note:
            Remember to avoid exceptions, such as division by zero.
        """
        return {
            # 'Metric1': Value,
            # 'Metric2': Value,
            # ...
        }

    # ─── Engine ───────────────────────────────────────────────────────────

    def evaluate(self):
        """Run a complete evaluation and save the output to a JSON file."""

        # ─── Initialize ───────────────────────────────────────────────

        start_time = time.time()
        logger.info(f"Start: {self.model.model_name} @ {self.__class__.__name__}")
        self.set_generation_configs()
        self.output_path = self.get_output_path()
        self.batched_dataset = self.load_batched_dataset()

        # ─── Metadata ─────────────────────────────────────────────────
        meta = {
            "evaluator": {
                "name": self.__class__.__name__,
                "num_batches": self.num_batches,
                "output_path": self.output_path,
                **self.more,
            },
            "model": {
                "name": self.model.model_name,
                "generation_configs": self.model.generation_configs,
                "loading_params": self.model.loading_params,
                "other_params": self.model.other_params,
            },
        }

        # ─── Evaluate ─────────────────────────────────────────────────

        results = []
        for idx, batch in enumerate(self.batched_dataset):
            logger.info(f"Batch {idx + 1}/{len(self.batched_dataset)}")
            results.extend(self.batch_scoring(batch))

        # ─── Postprocess ──────────────────────────────────────────────

        results = sorted(results, key=lambda x: x["id"])
        valid_results = self.remove_invalid_results(results)
        splitted_valid_results = self.split_results_by_type(valid_results)
        overall = self.compute_overall(valid_results) if len(valid_results) else {}
        overall_by_type = {
            type_name: self.compute_overall(sub)
            for type_name, sub in splitted_valid_results.items()
        }

        # ─── Save Output ──────────────────────────────────────────────

        output = {
            "meta": meta,
            "overall": overall,
            "overall_by_type": overall_by_type,
            "results": results,
        }
        self.save_output(output)

        # ─── Finish ───────────────────────────────────────────────────

        end_time = time.time()
        logger.info(f"Time elapsed: {end_time - start_time:.2f}s")
        logger.info(f"Finished: {self.output_path}\n")

    # ─── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def remove_invalid_results(results: list[dict]) -> list[dict]:
        """Remove invalid results from the list and return the cleaned results."""
        return [result for result in results if result["valid"]]

    @staticmethod
    def split_results_by_type(results: list[dict]) -> dict[str, list[dict]]:
        """Split the results by type and return a dictionary."""
        splitted = defaultdict(list)
        for result in results:
            type_name = result.get("data_point", {}).get("type", "default")
            splitted[type_name].append(result)
        return splitted

    def save_output(self, output: dict):
        """Save evaluation results."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


class DummyEvaluator(BaseEvaluator):
    """A dummy evaluator for testing purposes."""

    def set_generation_configs(self):
        new_configs = {
            "temperature": 0.1,
            "max_new_tokens": 24,
            "top_p": 0.9,
            "top_k": 5,
            "dummy_param": "dummy_value",
        }
        self.model.update_generation_configs(new_configs)

    def load_batched_dataset(self) -> list[list[dict]]:
        dataset = [
            {"id": 1, "type": "a", "q": "1+2-3*5", "a": -12},
            {"id": 2, "type": "b", "q": "3*4+33", "a": 45},
            {"id": 3, "type": "a", "q": "2*3-4", "a": 2},
            {"id": 4, "type": "c", "q": "3+4-5", "a": 2},
            {"id": 5, "type": "b", "q": "5*6*7", "a": 210},
        ]
        batches = [
            dataset[i : i + self.num_batches]
            for i in range(0, len(dataset), self.num_batches)
        ]
        return batches

    def scoring(self, data_point: dict) -> dict:
        evil_number = 0 if data_point["id"] % 2 else 1
        prediction = eval(data_point["q"]) + evil_number
        return {
            "metrics": {"correct": int(prediction == data_point["a"])},
            "log": {"prediction": prediction},
            "valid": True,
        }

    def compute_overall(self, results: list[dict]) -> dict:
        correct = sum(result["metrics"]["correct"] for result in results)
        total = len(results)
        if total == 0:
            return {"accuracy": 0}
        return {"accuracy": correct / total}
