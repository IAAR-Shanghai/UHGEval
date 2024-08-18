import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_llm import BaseLLM


class HuggingFace(BaseLLM):

    def __init__(
        self,
        model_name_or_path: str,
        generation_configs: dict = {},
        apply_chat_template: bool = False,
        system_message: str = "You are a helpful assistant.",
    ):
        """
        Args:
            model_name_or_path (str): The model name or path of the model.
            generation_configs (dict): The generation configurations.
            apply_chat_template (bool): Whether to apply chat template.
            system_message (str): If apply_chat_template is True, the system message to use.
        """
        super().__init__(
            model_name=model_name_or_path,
            generation_configs={
                "temperature": 1.0,
                "max_new_tokens": 256,
                "do_sample": True,
                **generation_configs,
            },
            loading_params={
                "loding_method": "huggingface",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            other_params={
                "apply_chat_template": apply_chat_template,
                "system_message": system_message,
            },
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def update_generation_configs(self, configs: dict):
        """Update the generation configurations."""
        if configs.get("temperature", -1) == 0:
            del configs["temperature"]
            configs["do_sample"] = False
        if "max_tokens" in configs:
            configs["max_new_tokens"] = configs["max_tokens"]
            del configs["max_tokens"]
        self.generation_configs.update(configs)

    def _request(self, query: str) -> str:
        if (
            self.other_params["apply_chat_template"]
            and self.tokenizer.chat_template is not None
        ):
            messages = []
            if self.other_params["system_message"] != "":
                messages.append(
                    {"role": "system", "content": self.other_params["system_message"]}
                )
            messages.append({"role": "user", "content": query})
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            input_ids = self.tokenizer.encode(query, return_tensors="pt").to(
                self.model.device
            )
        output_ids = self.model.generate(input_ids, **self.generation_configs)[0]
        response = self.tokenizer.decode(
            output_ids[len(input_ids[0]) - len(output_ids) :], skip_special_tokens=True
        )
        return response
