from typing import Callable

import evaluate  # Huggingface 包，可能需要代理以便网络访问正常
import jieba
from loguru import logger
from text2vec import Similarity

from utils.llm import Baichuan2_13B_Chat


def catch_all_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.warning(repr(e))
    return wrapper


@catch_all_exceptions
def bleu4_score(continuation:str, reference:str, with_penalty=False):
    f = lambda text: list(jieba.cut(text))
    bleu = evaluate.load('.cache/huggingface/bleu')
    results = bleu.compute(predictions=[continuation], references=[[reference]], tokenizer=f)
    score = results['bleu']
    brevity_penalty = results['brevity_penalty']
    if with_penalty:
        return score
    else:
        return 0.0 if brevity_penalty==0 else score/brevity_penalty


@catch_all_exceptions
def rougeL_score(continuation:str, reference:str):
    f = lambda text: list(jieba.cut(text))
    rouge = evaluate.load('.cache/huggingface/rouge')
    results = rouge.compute(predictions=[continuation], references=[[reference]], tokenizer=f, rouge_types=['rougeL'])
    score = results['rougeL']
    return score


@catch_all_exceptions
def kw_precision(continuation: str, reference: str, 
        kw_extracter: Callable[[str], list[str]] = Baichuan2_13B_Chat().extract_kws,         # TODO: 最后调整该项
        with_kw_list: bool = True
        )-> float | tuple[float, list[str], list[str]]:
    """使用model测度生成的续写sentence对原始新闻obj的合理性，分越高越合理"""
    kws = kw_extracter(continuation)
    if len(kws) == 0:
        return 0, [], [] if with_kw_list else 0
    appeared_kws = [kw for kw in kws if kw in reference]
    precision = len(appeared_kws) / len(kws)
    return precision, appeared_kws, kws if with_kw_list else precision


@catch_all_exceptions
def bert_score(continuation: str, reference: str) -> float:
    sim = Similarity()
    score = sim.get_score(continuation, reference)
    return score


def classifications(predictions:list[bool], references:list[bool]) -> tuple[float, float, float, float]:
    """二分类问题中计算accuracy, precision, recall 和 F1
    
    Args:
        predictions(list[bool]): 预测值列表，零一列表
        references(list[bool]): 真实值列表，零一列表
    """
    true_positive = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 1)
    false_positive = sum(1 for a, b in zip(references, predictions) if a == 0 and b == 1)
    false_negative = sum(1 for a, b in zip(references, predictions) if a == 1 and b == 0)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = sum(1 for a, b in zip(references, predictions) if a == b) / len(predictions) if len(predictions) > 0 else 0
    return accuracy, precision, recall, f1
