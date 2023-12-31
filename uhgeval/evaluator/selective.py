# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import datetime
import random
import re

from uhgeval.evaluator.base import BaseEvaluator
from uhgeval.llm.base import BaseLLM


class SelectiveEvaluator(BaseEvaluator):
    def __init__(self, model: BaseLLM, dataset: list[dict], output_dir: str = './output', seed = 22):
        super().__init__(model, dataset, output_dir)
        random.seed(seed)

    def set_model_params(self) -> None:
        params = {
            'temperature': 0.1,
            'max_new_tokens': 24,
            'top_p': 0.9,
            'top_k': 5,
        }
        self.model.update_params(**params)

    def scoring(self, data_point: dict) -> dict:
        swap = (random.random() > 0.5)

        contn1 = data_point['hallucinatedContinuation']
        contn2 = self._extract_first_sentence(data_point['newsRemainder'])
        if swap:
            contn1, contn2 = contn2, contn1  # Swap correct and incorrect sentences at random
        answer = self.model.compare_two_continuation(contn1, contn2, data_point)
        if swap:
            contn1, contn2 = contn2, contn1  # Swap back the two sentences
            answer = -answer + 3   # Swap back the answer, 1 becomes 2, and 2 becomes 1
        return {
            'metrics': {
                'correct': answer == 2
            },
            'log': {
                'swap': swap,
                'hallucinatedContinuation': contn1,
                'unhallucinatedContinuation': contn2,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': answer in [1, 2]
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            'accuracy': sum([result['metrics']['correct'] for result in results]) / len(results),
            'num': len(results)
        }

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        sentences = re.split(r'(?<=[。；？！])', text)
        return sentences[0]
