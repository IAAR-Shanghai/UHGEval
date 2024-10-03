import unittest

from eval_suite.llms.openai_api import OpenAIAPI


class TestOpenAIAPI(unittest.TestCase):
    def setUp(self):
        self.model_name = "Qwen/Qwen2-7B-Instruct"
        self.api_key = "your_api_key"
        self.base_url = "https://api.openai.com/v1/"
        self.model = OpenAIAPI(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def test_init(self):
        self.assertEqual(self.model.model_name, self.model_name)
        self.assertEqual(self.model.client.api_key, self.api_key)
        self.assertEqual(self.model.client.base_url, self.base_url)


if __name__ == "__main__":
    unittest.main()
