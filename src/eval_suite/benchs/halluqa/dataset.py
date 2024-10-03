import json
import os

from ..base_dataset import BaseDataset


class HalluQADataset(BaseDataset):
    def __init__(self, path: str):
        self.data = []
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        for item in self.data:
            item["id"] = item["question_id"]
            del item["question_id"]
            if "Category" in item:
                item["type"] = item["Category"]
                del item["Category"]

    def load(self) -> list[dict]:
        return self.data
