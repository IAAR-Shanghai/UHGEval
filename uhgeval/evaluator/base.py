# @Author : Shichao Song, YeZhaohui Wang
# @Email  : song.shichao@outlook.com, wyzh0912@126.com


import copy
import json
import os
from abc import ABC, abstractmethod

from loguru import logger
from tqdm import tqdm

from uhgeval.llm.base import BaseLLM


class BaseEvaluator(ABC):
    def __init__(self, model: BaseLLM,
                 dataset: list[dict], output_dir: str = './output'):
        """
        Args:
            model (BaseLLM): The large language model to be evaluated.
            dataset (list[dict]): The dataset for evaluation.
            output_dir (str): The directory for result output and caching.
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

    @abstractmethod
    def scoring(self, data_point: dict) -> dict:
        """Invoke self.model to evaluate data_point.

        Returns:
            dict: A result dictionary containing three mandatory fields: `metrics`, `log`, `valid`.
        """
        return {
            'metrics': {
                # Numerical results to be recorded by subclasses, mandatory.
                # Such as accuracy, recall, bleu, rouge, etc.
            },
            'log': {
                # String results to be recorded by subclasses, optional.
                # Such as model output.
            },
            'valid': ...
            # Boolean result to be recorded by subclasses, indicating whether the evaluation is valid, mandatory.
            # True or False.
        }

    def batch_scoring(self, dataset: list[dict], sort=True,
                      show_progress_bar=False, contain_original_data=False) -> list[dict]:
        """Perform batch scoring on the given dataset.

        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            list[dict]: List of results.
        """

        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        for data_point in (tqdm(
                dataset, desc=self.model.params['model_name']) if show_progress_bar else dataset):
            if data_point['id'] in saved_ids:
                continue  # Skip results that have already been evaluated and are valid
            try:
                result = {'id': data_point['id'], **self.scoring(data_point)}
                if contain_original_data:
                    result['original_data'] = data_point
                results.append(result)
            except Exception as e:
                logger.warning(repr(e))

        return sorted(results, key=lambda x: x['id']) if sort else results

    @abstractmethod
    def compute_overall(self, results: list[dict]) -> dict:
        """Extract and aggregate results from individual data points in the results.
        For example, calculate mean, variance, etc.

        Returns:
            dict: A result dictionary that can store any number and form of fields.
        """
        return {
            # 'Metric1': Value,
            # 'Metric2': Value,
            # ...
        }

    def save_output(self, output: dict) -> None:
        """Save evaluation results."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def read_output(self) -> dict:
        with open(self.output_path, encoding='utf-8') as f:
            return json.load(f)

    def run(self, sort=True, show_progress_bar=False,
            contain_original_data=True) -> dict:
        """Run a complete evaluation.

        Args:
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.
            contain_original_data (bool): Whether to include original data in the results for debugging.

        Returns:
            dict: Output dictionary contains fields such as: info, overall, results, etc.
        """
        info = {'evaluator': self.__class__.__name__,
                'llm': self.model.params['model_name']}

        self.set_model_params()
        results = self.batch_scoring(
            self.dataset,
            sort,
            show_progress_bar,
            contain_original_data)
        valid_results = self.remove_invalid(results)
      #  splitted_valid_results = self.split_results_by_type(valid_results)

        try:
            overall = self.compute_overall(
                valid_results) if len(valid_results) > 0 else {}
            # overall_by_type = {
            #     'overall-' + type_: (self.compute_overall(sub_valid_results) if len(sub_valid_results) > 0 else {})
            #     for type_, sub_valid_results in splitted_valid_results.items()
            # }
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()
            # overall_by_type = dict()

        self.save_output(
            output := {
                'info': info,
                'overall': overall,
                # **overall_by_type,
                'results': results})
        print(f'Output saved at {self.output_path}!')
        return output

    @staticmethod
    # def split_results_by_type(results: list[dict]) -> dict[str, list[dict]]:
    #     """Split results into four types based on 'doc', 'gen', 'kno', and 'num'."""
    #     return {
    #         'doc': [result for result in results if 'doc' in result['id']],
    #         'gen': [result for result in results if 'gen' in result['id']],
    #         'kno': [result for result in results if 'kno' in result['id']],
    #         'num': [result for result in results if 'num' in result['id']],
    #     }

    @staticmethod
    def remove_invalid(results: list[dict]) -> list[dict]:
        """Remove invalid results from the list and return the cleaned results."""
        return [result for result in results if result['valid']]
