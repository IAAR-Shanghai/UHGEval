import sys
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


if __name__ == '__main__':
    seed = 22
    enable_log_saving = True
    logger.remove()  # Remove all logger handlers including the stderr logger handler
    logger.add(sys.stderr, level=40)  # Update stderr logger
    logger.add('logs/uhgeval_{time}.log', level=0) if enable_log_saving else ...
    # TODO: Currently, loguru does not support log settings above when using the 'spawn' method in multiprocessing.

    dataset = XinhuaHallucinations('data/Xinhua/XinhuaHallucinations.json', shuffle=True, seed=seed).load()
    llms = [
        GPT(model_name='gpt-3.5-turbo', report=True), 
        GPT(model_name='gpt-4-0613', report=True), 
        GPT(model_name='gpt-4-1106-preview', report=True), 
    ]
    evaluators = [
        DiscriminativeEvaluatorKeywordLevel, 
        DiscriminativeEvaluatorSentenceLevel, 
        GenerativeEvaluator, 
        SelectiveEvaluator
    ]
    
    experiment_in_blocks(dataset, llms, evaluators, 3, 170, 0, seed)
    save_overalls()
    save_overalls_by_type()
