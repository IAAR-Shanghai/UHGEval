# @Author : Shichao Song
# @Email  : song.shichao@outlook.com
import multiprocessing as mp
import time
import yaml

from itertools import product
from loguru import logger
from uhgeval.evaluator.base import BaseEvaluator
from uhgeval.llm.base import BaseLLM


def experiment(
    dataset: list[dict],
    llms: list[BaseLLM],
    evaluators: list[BaseEvaluator],
    processes: int = 3,
    seed: int = 22
) -> None:
    """Run experiment with multiple language models and evaluators on a given dataset.

    Args:
        dataset (list[dict]): The input dataset as a list of dictionaries.
        llms (list[BaseLLM]): List of large language models to evaluate.
        evaluators (list[BaseEvaluator]): List of evaluators to use for each large language model.
        processes (int, optional): Number of processes to use for parallel execution. Defaults to 3.
        seed (int, optional): Seed for reproducibility. Defaults to 22.
    """
    start = time.time()
    if processes > 1:
        p = mp.Pool(processes)

    for llm, evaluator in product(llms, evaluators):
        instantiated_eval = evaluator(llm, dataset)
        # TODO: Redesign the `BaseEvaluator` to ensure `SelectiveEvaluator` can receive `seed`

        if processes > 1:
            p.apply_async(func=instantiated_eval.run, kwds={'show_progress_bar': True, 'contain_original_data': True})
        else:
            instantiated_eval.run(show_progress_bar=True, contain_original_data=True)

    if processes > 1:
        p.close()
        p.join()
    print('End of evaluations.')
    print(f'Time used: {time.time()-start}s.')


def experiment_in_blocks(
    dataset: list[dict],
    llms: list[BaseLLM],
    evaluators: list[BaseEvaluator],
    processes: int = 3,
    num_blocks: int = 170,
    start_block: int = 0,
    seed: int = 22
) -> None:
    if processes > 1 and mp.get_start_method() != 'spawn':
        mp.set_start_method('spawn', force=True)  # CUDA requires spawn method to launch multiple processes
    start = time.time()

    total = len(dataset)
    block_size = total // num_blocks
    
    for idx in range(start_block, num_blocks+1):  # Dividing may result in leftovers, so +1
        range_begin = idx * block_size
        range_end = range_begin + block_size
        print(f'Block {idx+1} / {num_blocks}: [{range_begin}, {range_end}]')
        experiment(dataset[range_begin : range_end], llms, evaluators, processes, seed)

    print(f'Total time used: {time.time()-start}s.')
    print(f'END')

def load_yaml_conf(yaml_path: str):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        logger.error(f"File '{yaml_path}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Unable to load YAML file '{yaml_path}': {e}")
