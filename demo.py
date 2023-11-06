"""
DEMO

在样本数据集（example.json）上对 GPT-3.5-Turbo 进行选择式测评（SelectiveEvaluator）
"""

from utils.dataset import XinhuaHallucinations
from utils.llm import GPT
from utils.evaluator import SelectiveEvaluator

seed = 22
llm = GPT
dataset = XinhuaHallucinations('data/example.json', shuffle=True, seed=seed).load()
evaluator = SelectiveEvaluator(llm(temperature=0.3, max_new_tokens=16), dataset, seed=seed)
evaluator.run(show_progress_bar=True, contain_original_data=True)
