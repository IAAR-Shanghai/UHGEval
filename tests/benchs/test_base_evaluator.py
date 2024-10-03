import os
import unittest
from unittest.mock import MagicMock

from eval_suite.benchs.base_evaluator import DummyEvaluator
from eval_suite.llms.base_llm import BaseLLM


class TestDummyEvaluator(unittest.TestCase):
    def setUp(self):
        # Mock the BaseLLM model
        self.mock_model = MagicMock(spec=BaseLLM)
        self.mock_model.model_name = "dummy_model"
        self.mock_model.generation_configs = {}
        self.mock_model.loading_params = {}
        self.mock_model.other_params = {}

        # Instantiate the DummyEvaluator with the mocked model
        self.evaluator = DummyEvaluator(
            model=self.mock_model, num_batches=2, output_dir="./output"
        )

    def test_set_generation_configs(self):
        self.evaluator.set_generation_configs()
        self.assertEqual(self.mock_model.update_generation_configs.call_count, 1)
        self.assertIn(
            "dummy_param", self.mock_model.update_generation_configs.call_args[0][0]
        )

    def test_get_output_path(self):
        output_path = self.evaluator.get_output_path()
        expected_filename = "dummy_model_DummyEvaluator_20240816123045.json"
        expected_path = os.path.join(self.evaluator.output_dir, expected_filename)
        self.assertTrue(output_path.startswith(expected_path[:-19]))
        self.assertEqual(len(output_path), len(expected_path))

    def test_load_batched_dataset(self):
        batches = self.evaluator.load_batched_dataset()
        self.assertEqual(
            len(batches), 3
        )  # 5 data points split into 3 batches of size 2
        self.assertEqual(len(batches[0]), 2)  # First batch should have 2 data points
        self.assertEqual(len(batches[-1]), 1)  # Last batch should have 1 data point

    def test_scoring(self):
        data_point = {"id": 1, "q": "1+2", "a": 3}
        result = self.evaluator.scoring(data_point)
        self.assertTrue(result["valid"])
        self.assertIn("correct", result["metrics"])
        self.assertIn("prediction", result["log"])

    def test_batch_scoring(self):
        data_items = [{"id": 1, "q": "1+2", "a": 3}, {"id": 2, "q": "3+4", "a": 7}]
        results = self.evaluator.batch_scoring(data_items)
        self.assertEqual(len(results), 2)
        self.assertIn("id", results[0])
        self.assertIn("data_point", results[0])
        self.assertIn("metrics", results[0])

    def test_compute_overall(self):
        results = [
            {"metrics": {"correct": 1}},
            {"metrics": {"correct": 0}},
            {"metrics": {"correct": 1}},
        ]
        overall = self.evaluator.compute_overall(results)
        self.assertIn("accuracy", overall)
        self.assertEqual(overall["accuracy"], 2 / 3)

    def test_evaluate(self):
        # Mock the save_output method to avoid file I/O during testing
        self.evaluator.save_output = MagicMock()

        # Run the evaluation process
        self.evaluator.evaluate()

        # Check if save_output was called
        self.evaluator.save_output.assert_called_once()

        # Check that the evaluation process produced a valid result
        results = self.evaluator.save_output.call_args[0][0]["results"]
        self.assertGreater(len(results), 0)
        self.assertIn("meta", self.evaluator.save_output.call_args[0][0])
        self.assertIn("overall", self.evaluator.save_output.call_args[0][0])

    def test_remove_invalid_results(self):
        results = [
            {"id": 1, "valid": True},
            {"id": 2, "valid": False},
            {"id": 3, "valid": True},
        ]
        cleaned_results = self.evaluator.remove_invalid_results(results)
        self.assertEqual(len(cleaned_results), 2)
        self.assertTrue(all(result["valid"] for result in cleaned_results))

    def test_split_results_by_type(self):
        results = [
            {"id": 1, "data_point": {"type": "type1"}},
            {"id": 2, "data_point": {"type": "type1"}},
            {"id": 3, "data_point": {"type": "type2"}},
        ]
        splitted_results = self.evaluator.split_results_by_type(results)
        self.assertEqual(len(splitted_results), 2)
        self.assertEqual(len(splitted_results["type1"]), 2)
        self.assertEqual(len(splitted_results["type2"]), 1)


if __name__ == "__main__":
    unittest.main()
