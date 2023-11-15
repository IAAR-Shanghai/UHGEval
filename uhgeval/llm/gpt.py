# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import openai
from loguru import logger

from uhgeval.llm.base import BaseLLM
from uhgeval.configs import real_config as conf


class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        self.token_consumed = 0

    def request(self, query: str) -> str:
        openai.api_key = conf.GPT_api_key
        res = openai.ChatCompletion.create(
            model = self.params['model_name'],
            messages = [{"role": "user","content": query}],
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
        )
        real_res = res["choices"][0]["message"]["content"]

        self.token_consumed += res['usage']['total_tokens']
        logger.info(f'GPT token consumed: {self.token_consumed}') if self.report else ()
        return real_res
