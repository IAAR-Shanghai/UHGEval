# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


"""对 GPT-3.5-Turbo 进行5个数据点上的选择式测评（SelectiveEvaluator）"""


from uhgeval.dataset.xinhua import XinhuaHallucinations
from uhgeval.evaluator.selective import SelectiveEvaluator
from uhgeval.interface.analyst import save_overalls
from uhgeval.llm.gpt import GPT


llm = GPT(temperature=0.3, max_new_tokens=16)
dataset = XinhuaHallucinations('data/XinhuaHallucinations.json', shuffle=True).load()
evaluator = SelectiveEvaluator(llm, dataset[:5])
evaluator.run(show_progress_bar=True, contain_original_data=True)
save_overalls()
