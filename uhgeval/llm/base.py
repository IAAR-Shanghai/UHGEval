# @Author : Shichao Song, Yezhaohui Wang
# @Email  : song.shichao@outlook.com, wyzh0912@126.com


import copy
import os
import re
import random
from abc import ABC, abstractmethod

from loguru import logger


class BaseLLM(ABC):

    # ─── 1. Interface Functions ───────────────────────────────────────────

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
        self.post_init()
    
    def post_init(self):
        """Post initialization method for subclasses.
        Normally, this method should initialize the model and tokenizer.
        """
        ...

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

    @staticmethod
    def _read_prompt_template(filename: str) -> str:
        path = os.path.join('uhgeval/prompts/', filename)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''

    # ─── 2. Prompt Engineering Functions ──────────────────────────────────

    # ─── 2.1 For XinhuaHallucination Dataset ──────────────────────────────

    def continue_writing(self, obj:dict) -> str:
        template = self._read_prompt_template('continue_writing.txt')
        query = template.format(f'《{obj["headLine"]}》\n{obj["broadcastDate"][:10]}\n{obj["newsBeginning"]}')
        res = self.safe_request(query)
        sentences = re.split(r'(?<=[。；？！])', res)
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

    # ─── 2.2 For TruthfulQA Dataset ───────────────────────────────────────

    def answer_MC1(self, obj: dict) -> int:
        """Answer a multiple choice question which has only one correct choice.

        Returns:
            int : 0 or 1(Answer right or wrong)
        """
        template = self._read_prompt_template('truthfulqa_mc1.txt')
        items_list = list(obj["mc1_targets"].items())
        random.shuffle(items_list)
        obj['mc1_targets'] = dict(items_list)
        optiontext = obj['mc1_targets']
        target_answer = ''
        text = ''
        for index, (key, value) in enumerate(optiontext.items(), start=1):
            option_letter = ord('A') + index - 1
            text = text + f"{option_letter}: {key}\n"
            if value == 1:
                target_answer = ord('A') + index - 1
        query = template.format(
            QuestionText=obj['question'],
            optionlist=text,
        )
        res = self.safe_request(query)
        if res == target_answer:
            return 1
        else:
            return 0

    def qa_judge(self, question, answer) -> str:
        """Judge whether the answer is correct or not."""
        template = self._read_prompt_template('qa_judge.txt')
        query = template.format(
            question=question,
            answer=answer
        )
        res = self.safe_request(query)
        res_reformatted = res.strip().split('\n')[0].split('.')[0].lower().strip()
        return res_reformatted

    def answer_Generation(self, obj: dict) -> str:
        """Given a question, generate a 1-2 sentence answer without prompt engineering.

        Returns:
            str : model's output
        """
        query = obj['Question']
        res = self.safe_request(query)
        return res

    # ─── 2.3 For HalluQA Dataset ──────────────────────────────────────────

    def answer_hallqa_mc(self, obj: dict) -> str:
        """Answer a multiple choice question which has only one correct choice."""
        template = self._read_prompt_template('halluqa_mc.txt')
        query = template.format(question=obj['question'])
        res = self.safe_request(query)
        res_reformatted = res.strip().split('\n')[0].split('.')[0].lower().strip()
        return res_reformatted

    # ─── 2.4 For HaluEval Dataset ─────────────────────────────────────────

    def is_summarization_hallucinated(self, document: str, summary: str) -> str:
        """Determine if a summary contains hallucination."""
        template = self._read_prompt_template('is_summarization_hallucinated.txt')
        query = template.format(document=document, summary=summary)
        res = self.safe_request(query)
        res_reformatted = res.strip().split('\n')[0].split('.')[0].lower().strip()
        return res_reformatted
