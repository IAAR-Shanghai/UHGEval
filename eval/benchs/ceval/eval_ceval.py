from typing import Literal

from ...llms import BaseLLM
from ..base_evaluator import BaseEvaluator
from .dataset import CEvalDataset
from .utils import get_subject_mapping

QA_TEMPLATE = """
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
答案：{answer}
"""

PROMPT_TEMPLATE = """以下是中国关于{discipline}考试的单项选择题，请选出其中的正确答案。
{qa_examples}
{qa_test}"""


CEVAL_HARD_DISCIPLINES = ",".join(
    [
        "advanced_mathematics",
        "discrete_mathematics",
        "probability_and_statistics",
        "college_chemistry",
        "college_physics",
        "high_school_mathematics",
        "high_school_chemistry",
        "high_school_physics",
    ]
)


class CEvalEvaluator(BaseEvaluator):

    def __init__(
        self,
        model: BaseLLM,
        num_batches: int = 1,
        output_dir: str = "./output",
        disciplines: str = CEVAL_HARD_DISCIPLINES,
        split: Literal["test", "val", "dev"] = "val",
        num_shots: int = 2,
    ):
        super().__init__(
            model,
            num_batches,
            output_dir,
            disciplines=disciplines,
            split=split,
            num_shots=num_shots,
        )

        self.split = split

        # ─── Get Valid Disciplines ────────────────────────────────────

        self.all_disciplines = set(get_subject_mapping().keys())
        if disciplines is None:
            self.disciplines = self.all_disciplines
        else:
            self.disciplines = set(disciplines.split(",")) & self.all_disciplines

        # ─── Load Examples For Few-shot Learning ──────────────────────

        if num_shots > 0:
            ds = CEvalDataset(self.disciplines, split="dev")
            self.discipline_examples = ds.load_as_dict_of_discipline(num_shots)
        else:
            self.discipline_examples = {}

    def set_generation_configs(self) -> None:
        new_configs = {"max_new_tokens": 16, "do_sample": False}
        self.model.update_generation_configs(new_configs)

    def load_batched_dataset(self) -> list[list[dict]]:
        dataset = CEvalDataset(self.disciplines, split=self.split)
        batches = dataset.to_batched(self.num_batches)
        return batches

    def qa_prompt(self, examples: list[dict]) -> str:
        prompt = "".join(
            QA_TEMPLATE.format(
                question=example["question"],
                choice_a=example["A"],
                choice_b=example["B"],
                choice_c=example["C"],
                choice_d=example["D"],
                answer=example["answer"],
            )
            for example in examples
        )
        return prompt

    def scoring(self, data_point: dict) -> dict:
        discipline = data_point["type"]
        query = PROMPT_TEMPLATE.format(
            discipline=get_subject_mapping()[discipline][1],  # Get the Chinese name
            qa_examples=self.qa_prompt(self.discipline_examples[discipline]),
            qa_test=self.qa_prompt([data_point]),
        )
        query = query.strip()[:-1]  # Remove the answer to be predicted
        response = self.model.safe_request(query)
        answer = response.strip().split("\n")[0].strip()  # Get the first line
        return {
            "metrics": {
                "correct": answer == data_point["answer"],
            },
            "log": {
                "answer": answer,
                "response": response,
                "query": query,
            },
            "valid": answer != "",
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
            "accuracy": sum([result["metrics"]["correct"] for result in results])
            / len(results),
            "num": len(results),
        }
