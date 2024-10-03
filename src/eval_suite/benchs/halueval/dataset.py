import json
import os

from ..base_dataset import BaseDataset


class HaluEvalDataset(BaseDataset):
    def __init__(self, path: str):
        self.data = []
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data_point = json.loads(line)
                data_point["id"] = idx + 1
                self.data.append(data_point)

    def load(self) -> list[dict]:
        return self.data
