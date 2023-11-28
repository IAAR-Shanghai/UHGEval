# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import sys
import argparse
from loguru import logger

from uhgeval.dataset.xinhua import XinhuaHallucinations
from uhgeval.evaluator.discriminative import (
    DiscriminativeEvaluatorKeywordLevel,
    DiscriminativeEvaluatorSentenceLevel
)
from uhgeval.evaluator.generative import GenerativeEvaluator
from uhgeval.evaluator.selective import SelectiveEvaluator
from uhgeval.core.analyst import save_overalls, save_overalls_by_type
from uhgeval.core.experiment import experiment_in_blocks
from uhgeval.llm.gpt import GPT
from uhgeval.llm.remote import (
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


def parse_args(arguments: str = None):
    parser = argparse.ArgumentParser(description='UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation')
    parser.add_argument('--seed', dest='seed', type=int, default=22, help='Random seed')
    parser.add_argument('--enable-log-saving', dest='enable_log_saving', default=False, action='store_true', help='Enable log saving')
    parser.add_argument('--dataset-path', dest='dataset_path', default='data/Xinhua/XinhuaHallucinations.json', help='Path to the dataset')
    parser.add_argument('--llms', dest='llms', nargs='+', default=['GPT'], help='List of LLMs to be evaluated')
    parser.add_argument('--evaluators', dest='evaluators', nargs='+', default=['DiscriminativeEvaluatorKeywordLevel', 'DiscriminativeEvaluatorSentenceLevel', 'GenerativeEvaluator', 'SelectiveEvaluator'], help='List of evaluators to use')
    parser.add_argument('--processes', dest='processes', type=int, default=3, help='Number of processes for the experiment')
    parser.add_argument('--num-blocks', dest='num_blocks', type=int, default=1700, help='Number of blocks for the experiment')
    parser.add_argument('--start-block', dest='start_block', type=int, default=0, help='Starting block number')
    parser.add_argument('--save-results', dest='save_results', default=True, action='store_true', help='Save experiment results')
    return parser.parse_args()
    # TODO: Currently, this script does not support initialize llm parameters

def run(args):
    logger.remove()  # Remove all logger handlers including the stderr logger handler
    logger.add(sys.stderr, level=40)  # Update stderr logger
    logger.add('logs/uhgeval_{time}.log', level=0) if args.enable_log_saving else ...
    # TODO: Currently, loguru does not support log settings above when using the 'spawn' method in multiprocessing.

    dataset = XinhuaHallucinations(args.dataset_path, shuffle=True, seed=args.seed).load()
    
    llms = []
    for llm_name in args.llms:
        llm = globals()[llm_name]()  # Instantiate LLMs
        llms.append(llm)
    
    evaluators = [globals()[evaluator_name] for evaluator_name in args.evaluators]

    experiment_in_blocks(dataset, llms, evaluators, args.processes, args.num_blocks, args.start_block, seed=args.seed)
    
    if args.save_results:
        save_overalls()
        save_overalls_by_type()


if __name__ == '__main__':
    args = parse_args()
    run(args)
