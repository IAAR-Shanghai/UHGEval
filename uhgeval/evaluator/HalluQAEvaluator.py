# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import datetime
import re

from uhgeval.evaluator.base import BaseEvaluator


class HalluQAMCEvaluator(BaseEvaluator):

    def set_model_params(self) -> None:
        params = {
            'temperature': 0,
            'max_new_tokens': 24,
        }
        self.model.update_params(**params)

    def scoring(self, data_point: dict) -> dict:
        output = self.model.answer_hallqa_mc(data_point)
        formatted_output = re.sub(r'[^\w\s]', ' ', output, flags=re.UNICODE).split()
        ground_truth = data_point['answer'][-1].lower()
        return {
            'metrics': {
                'correct': int(ground_truth in formatted_output),
            },
            'log': {
                'output': output,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': any([option in formatted_output for option in 'abcde'])
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            'accuracy': sum([result['metrics']['correct'] for result in results]) / len(results),
            'num': len(results)
        }
