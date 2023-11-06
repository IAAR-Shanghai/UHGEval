import datetime
import json
import os
import re
import random
from abc import ABC, abstractmethod

from loguru import logger
from tqdm import tqdm

from utils.llm import LanguageModel
from utils.metric import (
    bleu4_score, 
    rougeL_score, 
    kw_precision,
    bert_score,
    classifications,
)

class Evaluator(ABC):
    def __init__(self, model: LanguageModel, dataset: list[dict], output_dir: str = './output'):
        """
        Args:
            model(LanguageModel): 待评测的大模型
            dataset(list[dict]): 幻觉评测数据集
            output_dir(str): 结果输出的目录，缓存也在这里，允许断点继续评测
        """
        self.model = model
        self.dataset = dataset

        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        self.output_path = os.path.join(
            output_dir, f'{self.__class__.__name__}_{model.params["model_name"]}.json')

    @abstractmethod  # 由各个评测器子类实现
    def scoring(self, data_point:dict) -> dict:
        """调用self.model进行data_point上的评测

        Returns:
            dict: 一个结果字典，保存四个必选的字段，如下所示。
        """
        return {
            'metrics': {
                # 子类要记录的数值型结果，即各种指标的值，必选
            },
            'log': {
                # 子类要记录的字符型结果，即模型输出结果或数据点的上下文等，可选
            },
            'valid': False  # 子类要记录的布尔型结果，指示该条评测是否有效，必选
        }
    
    def batch_scoring(self, dataset:list[dict], sort = True, show_progress_bar = False, contain_original_data = False) -> list[dict]:
        """给定数据集，进行批量评分
        
        Args:
            dataset(list[dict]): 评测数据集
            sort(bool): 是否对结果按照文件名排序
            show_progress_bar(bool): 是否显示进度条
        """
        
        if os.path.exists(self.output_path):  # 实现断点继续评测功能
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        for data_point in (tqdm(dataset, desc=self.model.params['model_name']) if show_progress_bar else dataset):
            if data_point['id'] in saved_ids:
                continue  # 跳过已经评测过且评测有效的文件
            try:
                result = {'id': data_point['id'], **self.scoring(data_point)}
                if contain_original_data:
                    result['original_data'] = data_point
                results.append(result)
            except Exception as e:
                logger.warning(repr(e))
            
        return sorted(results, key=lambda x: x['id']) if sort else results
    
    @abstractmethod  # 由各个评测器子类实现
    def compute_overall(self, results: list[dict]) -> dict:
        """用于从results当中提取单个数据点的结果，并进行归约，例如“求平均值”“求方差”“求召回率”

        Returns:
            dict: 一个结果字典，可以保存任意数量任意形式的字段。
        """
        return {
            # '指标1': 值,
            # '指标2': 值,
            # ...
        }

    def save_output(self, output: dict):
        """保存评测结果"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    
    def read_output(self) -> dict:
        with open(self.output_path) as f:
            return json.load(f)

    def run(self, sort = True, show_progress_bar = False, contain_original_data = True):
        """运行一次完整的评测
        Args:
            sort(bool): 是否对保存文件中的结果进行按照文件名排序
            show_progress_bar(bool): 是否显示进度条
            contain_original_data(bool): 是否在结果中包含原始数据，以便调试
        Returns:
            output 字典
        """
        info = {'evaluator': self.__class__.__name__, 'llm': self.model.params['model_name']}

        results = self.batch_scoring(self.dataset, sort, show_progress_bar, contain_original_data)
        valid_results = self.remove_invalid(results)
        splitted_valid_results = self.split_results_by_type(valid_results)

        try:
            overall = self.compute_overall(valid_results) if len(valid_results) > 0 else {}
            overall_by_type = {
                'overall-'+ type_: (self.compute_overall(sub_valid_results) if len(sub_valid_results) > 0 else {})
                for type_, sub_valid_results in splitted_valid_results.items()
            }
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()
            overall_by_type = dict()

        self.save_output(output:={'info': info, 'overall': overall, **overall_by_type, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output

    @staticmethod
    def split_results_by_type(results: list[dict]) -> dict[str, list[dict]]:
        """将results按照type分为四份"""
        return {
            'doc': [result for result in results if 'doc' in result['id']],
            'gen': [result for result in results if 'gen' in result['id']],
            'kno': [result for result in results if 'kno' in result['id']],
            'num': [result for result in results if 'num' in result['id']],
        }
    
    @staticmethod
    def remove_invalid(results: list[dict]) -> list[dict]:
        """将results中无效的去除掉并返回去除后的results"""
        return [result for result in results if result['valid']]


class GenerativeEvaluator(Evaluator):
    def scoring(self, data_point: dict) -> dict:
        continuation = self.model.continue_writing(data_point)
        precision, _, kws = kw_precision(continuation, data_point['newsRemainder'])
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

    def compute_overall(self, results: list[dict]):
        overall = {'bleu-4': 0, 'rouge-L': 0, 'keywordsPrecision': 0, 'bertScore': 0, 'length': 0}
        for result in results:
            overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}
        overall = {f'avg. {key}': value / len(results) for key, value in overall.items()}
        overall['num'] = len(results)
        return overall


class DiscriminativeEvaluatorSentenceLevel(Evaluator):
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


class DiscriminativeEvaluatorKeywordLevel(Evaluator):
    def scoring(self, data_point: dict) -> dict:
        """true和positive用来形容有幻觉的，因为这里是评测检查出幻觉的能力"""
        kws = list(data_point['allKeywords'].keys())
        appeared_kws = data_point['appearedKeywords']
        unappeared_kws = [kw for kw in kws if kw not in appeared_kws]  # 后续测评时不考虑已经出现在原文的关键词
        
        # 真实值
        true = [kw for kw in unappeared_kws if data_point['allKeywords'][kw].startswith('不合理')]
        false = [kw for kw in unappeared_kws if data_point['allKeywords'][kw].startswith('合理')]
        num_each_side = min(len(true), len(false))  # 正负样例数量要相同
        true, false = true[:num_each_side], false[:num_each_side]

        # 预测值
        predictions = dict()
        for kw in true:
            predictions[kw] = (1, *self.model.is_kw_hallucinated(kw, data_point, with_reason=True))
        for kw in false:
            predictions[kw] = (0, *self.model.is_kw_hallucinated(kw, data_point, with_reason=True))
        # 最后字典的格式：`{'keyword': (ground_truth, prediction, reason), ...}`
        
        # 获取指标值
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


class SelectiveEvaluator(Evaluator):
    def __init__(self, model: LanguageModel, dataset: list[dict], output_dir: str = './output', seed = 22):
        super().__init__(model, dataset, output_dir)
        random.seed(seed)
    
    def scoring(self, data_point: dict) -> dict:
        swap = (random.random() > 0.5)

        contn1 = data_point['hallucinatedContinuation']
        contn2 = self._extract_first_sentence(data_point['newsRemainder'])
        if swap:
            contn1, contn2 = contn2, contn1  # 交换两个句子
        answer = self.model.compare_two_continuation(contn1, contn2, data_point)
        if swap:
            contn1, contn2 = contn2, contn1  # 交换回两个句子
            answer = -answer + 3  # 交换答案，1改为2，2改为1
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
