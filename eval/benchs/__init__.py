from .base_evaluator import BaseEvaluator
from .exampleqa.eval_exampleqa import ExampleQAEvaluator
from .halluqa.eval_halluqa_mc import HalluQAMCEvaluator
from .halueval.eval_halueval_dialog import HaluEvalDialogEvaluator
from .halueval.eval_halueval_qa import HaluEvalQAEvaluator
from .halueval.eval_halueval_summa import HaluEvalSummaEvaluator
from .uhgeval.eval_disc_keyword import UHGDiscKeywordEvaluator
from .uhgeval.eval_disc_sentence import UHGDiscSentenceEvaluator
from .uhgeval.eval_gene import UHGGenerativeEvaluator
from .uhgeval.eval_sele import UHGSelectiveEvaluator

# ! Register all evaluators here in alphabetical order.
__all__ = [
    # ExampleQA
    "ExampleQAEvaluator",
    # HalluQA
    "HalluQAMCEvaluator",
    # HaluEval
    "HaluEvalDialogEvaluator",
    "HaluEvalQAEvaluator",
    "HaluEvalSummaEvaluator",
    # UHGEval
    "UHGDiscKeywordEvaluator",
    "UHGDiscSentenceEvaluator",
    "UHGGenerativeEvaluator",
    "UHGSelectiveEvaluator",
]


def load_evaluator(evaluator_name: str) -> BaseEvaluator:
    """Load an evaluator class by its name.

    Args:
        evaluator_name (str): The class name of the evaluator.
    """
    return globals()[evaluator_name]


def get_all_evaluator_classes() -> list[BaseEvaluator]:
    """Get all evaluator classes. Except for the base evaluator class."""
    evaluator_classes = []
    for evaluator_name in __all__:
        cls = load_evaluator(evaluator_name)
        if cls is not BaseEvaluator and issubclass(cls, BaseEvaluator):
            evaluator_classes.append(cls)
    return evaluator_classes
