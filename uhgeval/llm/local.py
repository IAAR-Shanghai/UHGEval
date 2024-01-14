# @Author : YeZhaohui Wang
# @Email  : wyzh0912@126.com


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from uhgeval.configs import real_config as conf
from uhgeval.llm.base import BaseLLM


class Aquila_34B_Chat(BaseLLM):
    # For a more formal inference approach, please refer to https://huggingface.co/BAAI/AquilaChat2-34B.
    def request(self, query: str) -> str:
        local_path = conf.Aquila_local_path
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(input_ids, **gen_kwargs)[0]
        response = tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response


class Baichuan2_13B_Chat(BaseLLM):
    def request(self, query: str) -> str:
        local_path = conf.Baichuan2_13b_local_path
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(input_ids, **gen_kwargs)[0]
        response = tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        print(response)
        print('\n')
        return response


class ChatGLM3_6B_Chat(BaseLLM):
    def request(self, query: str) -> str:
        local_path = conf.ChatGLM3_local_path
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path, trust_remote_code=True, device='cuda')
        gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(input_ids, **gen_kwargs)[0]
        response = tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        print(response)
        return response


class Qwen_14B_Chat(BaseLLM):
    def request(self, query: str) -> str:
        local_path = conf.Qwen_local_path
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto",
                                                     trust_remote_code=True).eval()
        gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(input_ids, **gen_kwargs)[0]
        response = tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        print(response)
        return response


class InternLM_20B_Chat(BaseLLM):
    def request(self, query: str) -> str:
        local_path = conf.InternLM_local_path
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).cuda()
        gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
        input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
        output = model.generate(input_ids, **gen_kwargs)[0]
        response = tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        print(response)
        return response
