import json

import requests

from uhgeval.llm.base import BaseLLM
from uhgeval.configs import real_config as conf


class VllmModel(BaseLLM):
    def __init__(self):
        super().__init__()
        self.vllm_url = ''

    def _base_prompt_template(self) -> str:
        return "{query}"

    def request(self, query: str) -> str:
        url = self.url

        template = self._base_prompt_template()
        query = template.format(query=query)
        payload = json.dumps({
            "prompt": query,
            "temperature": self.params['temperature'],
            "max_tokens": self.params['max_new_tokens'],
            "n": 1,
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        })
        headers = {
            'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['text'][0].replace(query, '')  # VLLM will automatically append the query to the response, so here we remove it.
        return res


class PHI2(VllmModel):
    def __init__(self):
        super().__init__()
        self.url = conf.PHI2_vllm_url

    def _base_prompt_template(self) -> str:
        return """Q: {query}\A:"""

    def request(self, query: str) -> str:
        return super().request(query)
