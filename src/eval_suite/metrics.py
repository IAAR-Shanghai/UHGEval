import jieba
import text2vec
from loguru import logger
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def bleu_4(predication: str, reference: str, tokenizer=None) -> float:
    """Calculate the BLEU-4 score of the predication."""

    if tokenizer is None:
        tokenizer = lambda text: list(jieba.cut(text))
    list_predication = list(tokenizer(predication))
    list_reference = list(tokenizer(reference))
    score = sentence_bleu([list_reference], list_predication)
    return score


def rouge_l(prediction: str, reference: str, tokenizer=None) -> float:
    """Calculate the ROUGE-L score of the predication."""

    class Tokenizer:
        """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

        def __init__(self, tokenizer_func):
            self.tokenizer_func = tokenizer_func

        def tokenize(self, text):
            return self.tokenizer_func(text)

    if tokenizer is None:
        tokenizer = Tokenizer(lambda text: list(jieba.cut(text)))
    scorer = rouge_scorer.RougeScorer(rouge_types=["rougeL"], tokenizer=tokenizer)
    score = scorer.score(reference, prediction)["rougeL"].fmeasure
    return score


def keyword_precision(keywords: list[str], reference: str) -> float:
    """Measure the precision of keywords that appear in the reference."""
    if len(keywords) == 0:
        return 0
    appeared_kws = [kw for kw in keywords if kw in reference]
    precision = len(appeared_kws) / len(keywords)
    return precision


def bert_score(continuation: str, reference: str) -> float:
    logger.remove()
    sim = text2vec.Similarity("shibing624/text2vec-base-chinese")
    score = sim.get_score(continuation, reference)
    return score
