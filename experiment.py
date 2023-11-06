"""
一键进行实验

* 支持多个模型，在多评测器上，进行多进程的实验
* 支持分块，递增保存实验结果，以免因为意外错误丢失数据
"""

import multiprocessing as mp
import os
os.environ['HF_EVALUATE_OFFLINE'] = '1'  # 加载HuggingFace的evaluate模块需要联网加载，加载过后便可以启用此行
import time

from utils.dataset import XinhuaHallucinations
from utils.llm import (
    LanguageModel,
    Aquila_34B_Chat,
    Baichuan2_13B_Chat,
    Baichuan2_53B_Chat,
    ChatGLM2_6B_Chat,
    InternLM_20B_Chat,
    Xinyu_7B_Chat,
    Xinyu_70B_Chat,
    Qwen_14B_Chat,
    GPT_transit,
)
from utils.evaluator import (
    GenerativeEvaluator, 
    DiscriminativeEvaluatorSentenceLevel,
    DiscriminativeEvaluatorKeywordLevel,
    SelectiveEvaluator,
)


def experiment(dataset: list[dict], llms: list[LanguageModel], processes: int = 3, seed: int = 22) -> None:
    start = time.time()
    if processes > 1:
        p = mp.Pool(processes)

    for llm in llms:
        evaluators = [
            SelectiveEvaluator(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=24), dataset, seed=seed),
            GenerativeEvaluator(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=64), dataset),
            DiscriminativeEvaluatorKeywordLevel(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=24), dataset),
            DiscriminativeEvaluatorSentenceLevel(llm.update_params(inplace=False, temperature=0.1, max_new_tokens=24), dataset),  # TODO 实际使用时可以调低 max_new_tokens
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


def overalls2csv(output_dir: str = './output', filename: str = 'overalls.csv') -> None:
    import csv
    import json
    from collections import defaultdict

    # 读取所有 output 中的 overall
    overalls = defaultdict(lambda: defaultdict(dict))
    outputs = sorted(os.listdir(output_dir))
    outputs.remove(filename) if filename in outputs else ...
    for output in outputs:
        with open(os.path.join(output_dir, output)) as f:
            obj = json.load(f)
            llm, evaluator = obj['info']['llm'], obj['info']['evaluator']
            overalls[llm][evaluator] = obj['overall']
    
    # 提取列名
    evaluator_metric = []
    for obj in overalls.values():
        for evaluator, overall in obj.items():
            for metric in overall.keys():
                tmp = evaluator + ': ' + metric
                evaluator_metric.append(tmp) if tmp not in evaluator_metric else ...
    
    # 写入CSV文件
    csvfile = open(os.path.join(output_dir, filename), 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['LLM'] + evaluator_metric)
    for llm_name, obj in overalls.items():
        row = [llm_name]
        for item in evaluator_metric:
            evaluator, metric = item.split(': ')
            row.append(obj.get(evaluator, {}).get(metric, ''))
        writer.writerow(row)
    csvfile.close()

    print(f'Overall results saved at {output_dir}/{filename}')


if __name__ == '__main__':
    mp.set_start_method('spawn')  # CUDA 需要 spawn 方式来启动多进程
    dataset = XinhuaHallucinations('data/XinhuaHallucinations.json', shuffle=True, seed=22).load()
    llms = [
        GPT_transit(model_name='gpt-4-0613', report=True), 
        Aquila_34B_Chat(), 
        GPT_transit(model_name='gpt-3.5-turbo', report=True), 
        ChatGLM2_6B_Chat(), 
        Xinyu_70B_Chat(), 
        Baichuan2_13B_Chat(), 
        Baichuan2_53B_Chat(), 
        Xinyu_7B_Chat(), 
        Qwen_14B_Chat(), 
        InternLM_20B_Chat()
    ]
    
    start = time.time()
    blocks = 170
    total = len(dataset)
    block_size = total // blocks
    for idx in range(blocks+1):  # 整除可能会导致最后还有一些没有包含，所以+1
        # 分块保存，以免因为各种意外问题中断，导致所有数据丢失
        block_start = idx * block_size
        print(f'Block {idx+1} / {blocks}: [{block_start}, {block_start+block_size}]')
        experiment(dataset[block_start : block_start+block_size], llms[:], 3)
        overalls2csv()
    print(f'Total time used: {time.time()-start}s.')
    print(f'END')
