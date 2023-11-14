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
    dataset = XinhuaHallucinations('data/XinhuaHallucinations.json', shuffle=True, seed=22).load()
    llms = [
        InternLM_20B_Chat(), 
        GPT_transit(model_name='gpt-3.5-turbo', report=True), 
    ]
    evaluators = [DiscriminativeEvaluatorKeywordLevel, DiscriminativeEvaluatorSentenceLevel, GenerativeEvaluator, SelectiveEvaluator]
    experiment_in_blocks(dataset, llms, evaluators, 3, 1700, 0, 22)
