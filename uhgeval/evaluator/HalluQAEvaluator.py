# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import datetime

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
        ground_truth = data_point['answer'][-1].lower()
        return {
            'metrics': {
                'correct': int(output==ground_truth),
            },
            'log': {
                'output': output,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': output in 'ABCDE'
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            'accuracy': sum([result['metrics']['correct'] for result in results]) / len(results),
            'num': len(results)
        }
