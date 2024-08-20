from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..logging import logger
from .base_llm import BaseLLM


class OpenAIAPI(BaseLLM):

    def __init__(
        self,
        model_name: str,
        generation_configs: dict = {},
        api_key: str = None,
        base_url: str = None,
    ):
        """
        Args:
            model_name (str): The model name.
            generation_configs (dict): The generation configurations.
            api_key (str): The API key. If None, the environment variable "OPENAI_API_KEY" will be used.
            base_url (str): The base URL. If None, "https://api.openai.com/v1" will be used.
        """
        super().__init__(
            model_name=model_name,
            generation_configs={
                "temperature": 1.0,
                "max_tokens": 256,
                **generation_configs,
            },
            loading_params=(
                {"loading_method": "openai_api", "base_url": base_url}
                if base_url
                else {"loading_method": "openai_api"}
            ),
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def update_generation_configs(self, configs: dict):
        """Update the generation configurations."""
        if "max_new_tokens" in configs:
            configs["max_tokens"] = configs["max_new_tokens"]
            del configs["max_new_tokens"]
        if "top_k" in configs:
            del configs["top_k"]
        if "do_sample" in configs:
            if configs["do_sample"] == False:
                configs["temperature"] = 0.0
            del configs["do_sample"]
        self.generation_configs.update(configs)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
    def _request(self, query: str) -> str:
        response_obj = self.client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model=self.model_name,
            **self.generation_configs,
        )
        response = response_obj.choices[0].message.content
        logger.debug(f"{self.model_name} consumed: {response_obj.usage.json()}")
        return response
