from abc import ABC, abstractmethod

from ..logging import logger


class BaseLLM(ABC):

    def __init__(
        self,
        model_name: str,
        generation_configs: dict = {},
        loading_params: dict = {},
        other_params: dict = {},
    ):
        """
        Args:
            model_name (str): The name of the language model.
            generation_configs (dict): The configurations for text generation.
            loading_params (dict): The parameters for loading the language model.
            other_params (dict): Other parameters.
        """
        self.model_name = model_name
        self.generation_configs = generation_configs
        self.loading_params = loading_params
        self.other_params = other_params

    def update_generation_configs(self, configs: dict):
        """Update the generation configurations."""
        self.generation_configs.update(configs)

    @abstractmethod
    def _request(self, query: str) -> str:
        """Make a request to the language model."""
        return ""

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            response = self._request(query)
        except Exception as e:
            logger.warning(f"An error occurred: {e}")
            response = ""
        if response.startswith(query):  # Remove the query from the response
            response = response[len(query) :]
        return response
