# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import json
import re
import requests

from loguru import logger

from uhgeval.llm.base import BaseLLM
from uhgeval.configs import real_config as conf


class Aquila_34B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.Aquila_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": float(self.params['temperature']),
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Aquila_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices']
        return res

    def continue_writing(self, obj: dict) -> str:
        return super()._continue_writing_without_instruction(self, obj)


class Baichuan2_13B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.Baichuan2_13B_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Baichuan2_13B_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class ChatGLM2_6B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.ChatGLM2_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.ChatGLM2_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class GPT_transit(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    def request(self, query: str) -> str:
        url = conf.GPT_transit_url
        payload = json.dumps({
            "model": self.params['model_name'],
            "messages": [{"role": "user", "content": query}],
            "temperature": self.params['temperature'],
            'max_tokens': self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
        })
        headers = {
            'token': conf.GPT_transit_token,
            'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()
        real_res = res["choices"][0]["message"]["content"]

        token_consumed = res['usage']['total_tokens']
        logger.info(f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res


class InternLM_20B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.InternLM_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.InternLM_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res


class Qwen_14B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.Qwen_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Qwen_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res

    def continue_writing(self, obj: dict) -> str:
        return super()._continue_writing_without_instruction(self, obj)


class Xinyu_7B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.Xinyu_7B_url
        payload = json.dumps({
            "prompt": query,
            "params": {
                "temperature": self.params['temperature'],
                "do_sample": True,
                "max_new_tokens": self.params['max_new_tokens'],
                "num_return_sequences": 1,
                "top_p": self.params['top_p'],
                "top_k": self.params['top_k'],
            }
        })
        headers = {
        'token': conf.Xinyu_7B_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['choices'][0]
        return res

    def continue_writing(self, obj:dict) -> str:
        template = "Human: 【生成任务：文本续写】我要你担任新闻编辑。我将为您提供与新闻相关的故事或主题，您将续写一篇评论文章，对已有文本进行符合逻辑的续写。您应该利用自己的经验，深思熟虑地解释为什么某事很重要，用事实支持主张，并补充已有故事中可能缺少的逻辑段落。\n请对以下文本进行续写。\n {} Assistant:"
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split('Assistant:')[-1].split('</s>')[0].strip()
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]


class Xinyu_70B_Chat(BaseLLM):
    def request(self, query) -> str:
        url = conf.Xinyu_70B_url
        payload = json.dumps({
            "prompt": query,
            "temperature": self.params['temperature'],
            "max_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        })
        headers = {
        'token': conf.Xinyu_70B_token,
        'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['text'][0]
        return res

    def continue_writing(self, obj:dict) -> str:
        template = "Human: 【生成任务：文本续写】我要你担任新闻编辑。我将为您提供与新闻相关的故事或主题，您将续写一篇评论文章，对已有文本进行符合逻辑的续写。您应该利用自己的经验，深思熟虑地解释为什么某事很重要，用事实支持主张，并补充已有故事中可能缺少的逻辑段落。\n请对以下文本进行续写。\n {} Assistant:"
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split('Assistant:')[-1].split('</s>')[0].strip()
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]
