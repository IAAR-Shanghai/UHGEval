# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


"""Evaluate GPT-3.5-Turbo on selective evaluation with 5 data points."""


from uhgeval.dataset.xinhua import XinhuaHallucinations
from uhgeval.evaluator.selective import SelectiveEvaluator
from uhgeval.interface.analyst import save_overalls
from uhgeval.llm.gpt import GPT


llm = GPT(model_name='gpt-3.5-turbo', temperature=0.3, max_new_tokens=16)
dataset = XinhuaHallucinations('data/XinhuaHallucinations.json', shuffle=True).load()
evaluator = SelectiveEvaluator(llm, dataset[:5])
evaluator.run(show_progress_bar=True, contain_original_data=True)
save_overalls()
