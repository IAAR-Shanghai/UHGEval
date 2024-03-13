import json

import requests

from uhgeval.llm.base import BaseLLM
from uhgeval.configs import real_config as conf


class VllmModel(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            "stop": ["<|endoftext|>", "<|im_end|>"],
        })
        headers = {
            'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['text']
        if isinstance(res, list):
            res = res[0]
        res = res.replace(query, '')  # VLLM will automatically append the query to the response, so here we remove it.
        return res


class BloomZ_3B(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.BloomZ_3B_vllm_url

    def _base_prompt_template(self) -> str:
        return """{query}"""


class Gemma_2B_Chat(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.Gemma_2B_Chat_vllm_url

    def _base_prompt_template(self) -> str:
        template = """<bos><start_of_turn>user""" \
            """{query}<end_of_turn>""" \
            """<start_of_turn>model"""
        return template


class InternLM2_1_8B_Chat(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.InternLM2_1_8B_Chat_vllm_url

    def _base_prompt_template(self) -> str:
        template = """<|im_start|>system""" \
            """You are a helpful assistant.<|im_end|>""" \
            """<|im_start|>user""" \
            """{query}<|im_end|>""" \
            """<|im_start|>assistant\n"""
        return template


class LLaMA2_13B_Chat(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.LLaMA2_13B_Chat_vllm_url

    def _base_prompt_template(self) -> str:
        template = """<s>[INST] <<SYS>>""" \
            """You are being tested. Follow the instruction below. """ \
            """<</SYS>> {query} [/INST] Sure, I'd be happy to help. Here is the answer:"""
        return template


class LLaMA2_70B_Chat(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.LLaMA2_70B_Chat_vllm_url

    def _base_prompt_template(self) -> str:
        template = """<s>[INST] <<SYS>>""" \
            """You are being tested. Follow the instruction below. """ \
            """<</SYS>> {query} [/INST] Sure, I'd be happy to help. Here is the answer:"""
        return template


class LLaMA2_7B_Chat(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.LLaMA2_7B_Chat_vllm_url

    def _base_prompt_template(self) -> str:
        template = """<s>[INST] <<SYS>>""" \
            """You are being tested. Follow the instruction below. """ \
            """<</SYS>> {query} [/INST] Sure, I'd be happy to help. Here is the answer:"""
        return template


class NewModel(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.NewModel_vllm_url

    def _base_prompt_template(self) -> str:
        return """{query}"""


class OPT(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.OPT_vllm_url

    def _base_prompt_template(self) -> str:
        return """{query}"""


class PHI2(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.PHI2_vllm_url

    def _base_prompt_template(self) -> str:
        return """{query}"""


class Qwen1_5_4B_Chat(VllmModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = conf.Qwen1_5_4B_Chat_vllm_url

    def _base_prompt_template(self) -> str:
        template = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n""" \
                   """{query}<|im_end|>\n<|im_start|>assistant\n"""
        return template
