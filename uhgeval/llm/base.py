# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import copy
import os
import re
from abc import ABC, abstractmethod

from loguru import logger


class BaseLLM(ABC):
    def __init__(
            self, 
            model_name: str = None, 
            temperature: float = 1.0, 
            max_new_tokens: int = 1024, 
            top_p: float = 0.9,
            top_k: int = 5,
            **more_params
        ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }

    def update_params(self, inplace: bool = True, **params):
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    @abstractmethod
    def request(self, query:str) -> str:
        return ''

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            response = self.request(query)
        except Exception as e:
            logger.warning(repr(e))
            response = ''
        return response

    def continue_writing(self, obj:dict) -> str:
        template = self._read_prompt_template('continue_writing.txt')
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"][:10]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]

    @staticmethod
    def _continue_writing_without_instruction(self, obj:dict) -> str:
        """Generate a continuation without prompt engineering."""
        template = "{}"
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        real_res = res.split(query)[-1] if query in res else res
        real_res = real_res.replace('<s>', '').replace('</s>', '').strip()
        sentences = re.split(r'(?<=[。；？！])', real_res)
        return sentences[0]

    def extract_kws(self, sentence:str) -> list[str]:
        template = self._read_prompt_template('extract_kws.txt')
        query = template.format(sentence)
        res = self.safe_request(query)
        kws = res.split('<keywords>')[-1].split('</keywords>')[0].split('\n')
        filtered = [
            s.strip() 
            for s in kws 
            if s.strip() and s.strip() in sentence
        ]
        return filtered

    def is_kw_hallucinated(self, kw:str, obj:dict, with_reason: bool = False) -> int | tuple[int, str]:
        """Determine if a keyword exists as a hallucination.

        Returns:
            int or tuple: 0 or 1 (does not contain hallucination or contains hallucination); -1 if there is an error.
                If with_reason is True, return a tuple with the reason.
        """

        template = self._read_prompt_template('is_kw_hallucinated.txt')        
        query = template.format(
            headLine=obj['headLine'],
            broadcastDate=obj['broadcastDate'],
            newsBeginning=obj['newsBeginning'],
            continuation=obj['hallucinatedContinuation'],
            keyword=kw
        )
        res = self.safe_request(query)
        real_res = res.split(query)[-1]  # Remove repetition
        if real_res.startswith('不符合现实'):
            answer = 1
        elif real_res.startswith('符合现实'):
            answer = 0
        else:
            answer = -1
        return (answer, real_res.split('。')[0]) if with_reason else answer

    def compare_two_continuation(self, contn1: str, contn2: str, obj: dict) -> int:
        """Compare two continuations and determine which one is better.

        Returns:
            int: 1 or 2 (continuation 1 or continuation 2); -1 if there is an error.
        """

        template = self._read_prompt_template('compare_two_continuation.txt')
        query = template.format(
            headLine = obj['headLine'],
            broadcastDate = obj['broadcastDate'],
            newsBeginning = obj['newsBeginning'],
            contn1 = contn1,
            contn2 = contn2,
        )
        res = self.safe_request(query)
        real_res = res.split(query)[-1]  # Remove repetition
        real_res = real_res.split('更符合现实，更准确')[0].strip()  # Extract answer
        if real_res == 'A':
            answer = 1
        elif real_res == 'B':
            answer = 2
        else:
            answer = -1
        return answer

    def is_continuation_hallucinated(self, continuation:str, obj:dict, with_reason: bool = False) -> int | tuple[int, str]:
        """Determine if a continuation contains hallucination.

        Returns:
            int or tuple: 0 or 1 (does not contain hallucination or contains hallucination); -1 if there is an error.
                If with_reason is True, return a tuple with the reason.
        """

        template = self._read_prompt_template('is_continuation_hallucinated.txt')
        query = template.format(
            headLine = obj['headLine'],
            broadcastDate = obj['broadcastDate'],
            newsBeginning = obj['newsBeginning'],
            continuation = continuation
        )
        res = self.safe_request(query)
        real_res = res.split(query)[-1]  # Remove repetition
        if real_res.startswith('续写不符合现实'):
            answer = 1
        elif real_res.startswith('续写符合现实'):
            answer = 0
        else:
            answer = -1
        return (answer, real_res.split('。')[0]) if with_reason else answer

    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join('uhgeval/prompts/', filename)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''
