import unittest
from unittest.mock import MagicMock

from eval_suite.llms.base_llm import BaseLLM


class TestBaseLLM(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock(spec=BaseLLM)
        self.model.model_name = "LLM Instance"
        self.model.generation_configs = {}
        self.model.loading_params = {}
        self.model.other_params = {}

    def test_init(self):
        self.assertEqual(self.model.model_name, "LLM Instance")
        self.assertEqual(self.model.generation_configs, {})
        self.assertEqual(self.model.loading_params, {})
        self.assertEqual(self.model.other_params, {})

    def test_update_generation_configs(self):
        new_configs = {"temperature": 0.8, "top_p": 0.3}
        self.model.update_generation_configs(new_configs)
        self.model.update_generation_configs.assert_called_once_with(new_configs)


if __name__ == "__main__":
    unittest.main()
