# @Author : Shichao Song
# @Email  : song.shichao@outlook.com

import datetime

from uhgeval.evaluator.base import BaseEvaluator
from uhgeval.metric.common import (
    bleu4_score, 
    rougeL_score, 
    kw_precision,
    bert_score,
)

class GenerativeEvaluator(BaseEvaluator):
    def set_model_params(self) -> None:
        params = {
            'temperature': 0.1,
            'max_new_tokens': 64,
            'top_p': 0.9,
            'top_k': 5,
        }
        self.model.update_params(**params)

    def scoring(self, data_point: dict) -> dict:
        continuation = self.model.continue_writing(data_point)
        precision, _, kws = kw_precision(continuation, data_point['newsRemainder'], self.model.extract_kws)
        return {
            'metrics': {
                'bleu-4': bleu4_score(continuation, data_point['newsRemainder']) or 0.0,
                'rouge-L': rougeL_score(continuation, data_point['newsRemainder']) or 0.0,
                'keywordsPrecision': precision or 0.0,
                'bertScore': bert_score(continuation, data_point['newsRemainder']) or 0.0,
                'length': len(continuation)
            },
            'log': {
                'continuation': continuation,
                'keywords': kws,
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': len(continuation.strip()) != 0
        }

    def compute_overall(self, results: list[dict]) -> dict:
        overall = {'bleu-4': 0, 'rouge-L': 0, 'keywordsPrecision': 0, 'bertScore': 0, 'length': 0}
        for result in results:
            overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}
        overall = {f'avg. {key}': value / len(results) for key, value in overall.items()}
        overall['num'] = len(results)
        return overall
