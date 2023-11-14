# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import multiprocessing as mp
import time

from uhgeval.llm.base import BaseLLM
from uhgeval.evaluator.base import BaseEvaluator
from uhgeval.evaluator.discriminative import (
    DiscriminativeEvaluatorKeywordLevel,
    DiscriminativeEvaluatorSentenceLevel
)
from uhgeval.evaluator.generative import GenerativeEvaluator
from uhgeval.evaluator.selective import SelectiveEvaluator


def experiment(
    dataset: list[dict],
    llms: list[BaseLLM],
    evaluators: list[BaseEvaluator],
    processes: int = 3,
    seed: int = 22
) -> None:
    start = time.time()
    if processes > 1:
        p = mp.Pool(processes)

    for llm in llms:
        evaluators = [
            SelectiveEvaluator(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=24), dataset, seed=seed),
            GenerativeEvaluator(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=64), dataset),
            DiscriminativeEvaluatorKeywordLevel(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=24), dataset),
            DiscriminativeEvaluatorSentenceLevel(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=24), dataset),
        ]
        for evaluator in evaluators:
            (
                p.apply_async(func=evaluator.run, kwds={'show_progress_bar': True, 'contain_original_data': True})  # 仅当进程数大于 1 时，才进行多进程
                if processes > 1 else
                evaluator.run(show_progress_bar=True, contain_original_data=True)
            )

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
    if processes > 1:
        mp.set_start_method('spawn')  # CUDA requires spawn method to launch multiple processes
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
