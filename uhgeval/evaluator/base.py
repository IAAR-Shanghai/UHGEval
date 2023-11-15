# @Author : Shichao Song, Zhaohui Wangye
# @Email  : song.shichao@outlook.com, wyzh0912@126.com


import copy
import json
import os
from abc import ABC, abstractmethod

from loguru import logger
from tqdm import tqdm

from uhgeval.llm.base import BaseLLM


class BaseEvaluator(ABC):
    def __init__(self, model: BaseLLM, dataset: list[dict], output_dir: str = './output'):
        """
        Args:
            model(LanguageModel): 待评测的大模型
            dataset(list[dict]): 幻觉评测数据集
            output_dir(str): 结果输出的目录，缓存也在这里，允许断点继续评测
        """
        self.model = copy.deepcopy(model)
        self.dataset = dataset

        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        self.output_path = os.path.join(
            output_dir, f'{self.__class__.__name__}_{model.params["model_name"]}.json')

    @abstractmethod
    def set_model_params(self) -> None:
        """Use unified parameters for the evaluation."""
        params = {
            # Subclass fill in the parameters here.
        }
        self.model.update_params(**params)

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

    def save_output(self, output: dict) -> None:
        """保存评测结果"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    
    def read_output(self) -> dict:
        with open(self.output_path) as f:
            return json.load(f)

    def run(self, sort = True, show_progress_bar = False, contain_original_data = True) -> dict:
        """运行一次完整的评测
        Args:
            sort(bool): 是否对保存文件中的结果进行按照文件名排序
            show_progress_bar(bool): 是否显示进度条
            contain_original_data(bool): 是否在结果中包含原始数据，以便调试
        Returns:
            output 字典
        """
        info = {'evaluator': self.__class__.__name__, 'llm': self.model.params['model_name']}

        self.set_model_params()
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
