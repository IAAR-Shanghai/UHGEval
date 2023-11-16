# @Author : Shichao Song, Zhaohui Wangye
# @Email  : song.shichao@outlook.com, wyzh0912@126.com


import datetime
import re

from uhgeval.evaluator.base import BaseEvaluator
from uhgeval.metric.common import classifications


class DiscriminativeEvaluatorSentenceLevel(BaseEvaluator):
    def set_model_params(self) -> None:
        params = {
            'temperature': 0.1,
            'max_new_tokens': 24,
            'top_p': 0.9,
            'top_k': 5,
        }
        self.model.update_params(**params)

    def scoring(self, data_point: dict) -> dict:
        hallu = data_point['hallucinatedContinuation']
        unhallu = self._extract_first_sentence(data_point['newsRemainder'])
        answer_hallu, reason_hallu = self.model.is_continuation_hallucinated(hallu, data_point, with_reason=True)
        answer_unhallu, reason_unhallu = self.model.is_continuation_hallucinated(unhallu, data_point, with_reason=True)

        return {
            'metrics': {
                'accuracy': ((answer_hallu==1)+(answer_unhallu==0)) / 2.0
            },
            'log': {
                'hallucinatedContinuation': hallu,
                'response_to_hallucinatedContinuation': reason_hallu,
                'unhallucinatedContinuation': unhallu,
                'response_to_unhallucinatedContinuation': reason_unhallu,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': answer_hallu in {0, 1} and answer_unhallu in {0, 1}
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            'avg. accuracy': sum([result['metrics']['accuracy'] for result in results]) / len(results),
            'num': len(results)
        }

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        sentences = re.split(r'(?<=[。；？！])', text)
        return sentences[0]


class DiscriminativeEvaluatorKeywordLevel(BaseEvaluator):
    def set_model_params(self) -> None:
        params = {
            'temperature': 0.1,
            'max_new_tokens': 24,
            'top_p': 0.9,
            'top_k': 5,
        }
        self.model.update_params(**params)

    def scoring(self, data_point: dict) -> dict:
        """Evaluate the keyword hallucination detection ability of the model.

        Args:
            data_point (dict): A data point in the dataset.

        Returns:
            dict: A result dictionary.
        
        Note:
            True and positive are used to describe hallucinations. 
            False and negative are used to describe non-hallucinations.
        """

        kws = list(data_point['allKeywords'].keys())
        appeared_kws = data_point['appearedKeywords']
        unappeared_kws = [kw for kw in kws if kw not in appeared_kws]
        # Do not consider keywords already appeared in the original text
        
        # Ground truth values
        true = [kw for kw in unappeared_kws if data_point['allKeywords'][kw].startswith('不合理')]
        false = [kw for kw in unappeared_kws if data_point['allKeywords'][kw].startswith('合理')]
        num_each_side = min(len(true), len(false))  # Equal number of positive and negative examples
        true, false = true[:num_each_side], false[:num_each_side]

        # Predicted values
        predictions = dict()
        for kw in true:
            predictions[kw] = (1, *self.model.is_kw_hallucinated(kw, data_point, with_reason=True))
        for kw in false:
            predictions[kw] = (0, *self.model.is_kw_hallucinated(kw, data_point, with_reason=True))
        # Dictionary format: `{'keyword': (ground_truth, prediction, reason), ...}`
        
        # Get metric values
        accuracy, precision, recall, f1 = classifications(
            predictions=[item[1] for item in predictions.values()],
            references=[item[0] for item in predictions.values()]
        )
        return {
            'metrics': {
                'accuracy': accuracy,
                'num_kws': num_each_side*2
           },
            'log': {
                'predictions': predictions,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': num_each_side > 0 and not any([answer not in {0, 1} for answer in [item[1] for item in predictions.values()]])
        }

    def compute_overall(self, results: list[dict]) -> dict:
        overall = {'accuracy': 0, 'num_kws': 0}
        for result in results:
            overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}
        overall = {f'avg. {key}': value / len(results) for key, value in overall.items()}
        overall['num'] = len(results)
        return overall
