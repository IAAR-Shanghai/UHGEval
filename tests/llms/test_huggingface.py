import unittest

import torch

from eval_suite.llms.huggingface import HuggingFace


class TestHuggingFace(unittest.TestCase):
    def setUp(self):
        self.model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
        self.generation_configs = {
            "temperature": 1.0,
            "max_new_tokens": 4,
            "do_sample": True,
        }
        self.model = HuggingFace(
            model_name_or_path=self.model_name_or_path,
            generation_configs=self.generation_configs,
            apply_chat_template=True,
        )

    def test_init(self):
        self.assertEqual(self.model.model_name, self.model_name_or_path)
        self.assertEqual(self.model.generation_configs["temperature"], 1.0)
        self.assertEqual(self.model.generation_configs["max_new_tokens"], 4)
        self.assertEqual(self.model.generation_configs["do_sample"], True)
        self.assertEqual(self.model.other_params["apply_chat_template"], True)
        self.assertEqual(
            self.model.other_params["system_message"], "You are a helpful assistant."
        )

    def test_real_request(self):
        query = "Hi."
        response = self.model._request(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main()
