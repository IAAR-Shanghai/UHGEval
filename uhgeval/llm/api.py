# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import json

import openai
import requests
from loguru import logger

from uhgeval.configs import real_config as conf
from uhgeval.llm.base import BaseLLM


class Baichuan2_53B_Chat(BaseLLM):
    def request(self, query) -> str:
        import time
        url = conf.Baichuan2_53B_url
        api_key = conf.Baichuan2_53B_api_key
        secret_key = conf.Baichuan2_53B_secret_key
        time_stamp = int(time.time())

        json_data = json.dumps({
            "model": "Baichuan2-53B",
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "parameters": {
                "temperature": self.params['temperature'],
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        def _calculate_md5(input_string):
            import hashlib
            md5 = hashlib.md5()
            md5.update(input_string.encode('utf-8'))
            encrypted = md5.hexdigest()
            return encrypted
        signature = _calculate_md5(secret_key + json_data + str(time_stamp))
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }
        res = requests.post(url, data=json_data, headers=headers)
        res = res.json()['data']['messages'][0]['content']
        return res


class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    def request(self, query: str) -> str:
        openai.api_key = conf.GPT_api_key
        if conf.GPT_api_base and conf.GPT_api_base.strip():
            openai.base_url = conf.GPT_api_base
        res = openai.ChatCompletion.create(
            model = self.params['model_name'],
            messages = [{"role": "user","content": query}],
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
        )
        real_res = res["choices"][0]["message"]["content"]

        token_consumed = res['usage']['total_tokens']
        logger.info(f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res
