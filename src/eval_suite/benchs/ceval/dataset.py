from collections import defaultdict
from typing import Literal

from datasets import load_dataset
from tqdm import tqdm

from ..base_dataset import BaseDataset
from .utils import get_subject_mapping


class CEvalDataset(BaseDataset):
    def __init__(
        self, disciplines: set[str] = None, split: Literal["test", "val", "dev"] = "val"
    ):
        """
        Args:
            disciplines: Disciplines to load. If None, all disciplines will be loaded.
            split: The split to load. One of "test", "val", "dev".
        """
        subject_mapping = get_subject_mapping()
        self.data = []
        if disciplines is None:
            disciplines = set(subject_mapping.keys())

        for discipline in tqdm(disciplines, desc=f"Loading CEval > {split}"):
            ds = load_dataset("ceval/ceval-exam", discipline, split=split)
            for item in ds:
                item["id"] = f"{discipline}_{split}_{item['id']:>04}"
                item["type"] = discipline
                self.data.append(item)

    def load(self) -> list[dict]:
        return self.data

    def load_as_dict_of_discipline(self, num_shots: int) -> dict[str, list[dict]]:
        examples = defaultdict(list)
        for item in self.data:
            if len(examples[item["type"]]) < num_shots:
                examples[item["type"]].append(item)
        return examples
