# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import datetime
import random

from uhgeval.evaluator.base import BaseEvaluator
from uhgeval.llm.base import BaseLLM


class HaluEvalSummarizationEvaluator(BaseEvaluator):

    def __init__(self, model: BaseLLM, dataset: list[dict], output_dir: str = './output', seed = 22):
        super().__init__(model, dataset, output_dir)
        self.seed = seed

    def set_model_params(self) -> None:
        params = {
            'temperature': 0,
            'max_new_tokens': 24,
        }
        self.model.update_params(**params)

    def scoring(self, data_point: dict) -> dict:
        document = data_point['document']
        random.seed(self.seed+len(document))
        choose_right = int(random.random() > 0.5)
        summary = data_point['right_summary'] if choose_right else data_point['hallucinated_summary']
        output = self.model.is_summarization_hallucinated(document, summary)
        processed_output = output.lower().replace('.', '').replace(',', '').split()
        if 'yes' in processed_output:
            model_answer = 1
        elif 'no' in processed_output:
            model_answer = 0
        else:
            model_answer = -1
        return {
            'metrics': {
                'correct': int(model_answer==choose_right),
            },
            'log': {
                'output': output,
                'choose_right': choose_right,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': model_answer in [0, 1]
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            'accuracy': sum([result['metrics']['correct'] for result in results]) / len(results),
            'num': len(results)
        }
