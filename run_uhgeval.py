import sys
from loguru import logger

from uhgeval.dataset.xinhua import XinhuaHallucinations
from uhgeval.evaluator.discriminative import (
    DiscriminativeEvaluatorKeywordLevel,
    DiscriminativeEvaluatorSentenceLevel
)
from uhgeval.evaluator.generative import GenerativeEvaluator
from uhgeval.evaluator.selective import SelectiveEvaluator
from uhgeval.interface.experiment import experiment_in_blocks
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
    enable_log_saving = True
    logger.remove()  # Remove all logger handlers including the stderr logger handler
    logger.add(sys.stderr, level=40)  # Update stderr logger
    logger.add('logs/uhgeval_{time}.log', level=0) if enable_log_saving else ...
    # TODO: Currently, loguru does not support log settings above when using the 'spawn' method in multiprocessing.
    
    dataset = XinhuaHallucinations('data/XinhuaHallucinations.json', shuffle=True, seed=22).load()
    llms = [
        InternLM_20B_Chat(), 
        GPT_transit(model_name='gpt-3.5-turbo', report=True), 
    ]
    evaluators = [DiscriminativeEvaluatorKeywordLevel, DiscriminativeEvaluatorSentenceLevel, GenerativeEvaluator, SelectiveEvaluator]
    experiment_in_blocks(dataset, llms, evaluators, 3, 1700, 0, 22)
