import json
import os

from ..base_dataset import BaseDataset


class UHGEvalDataset(BaseDataset):
    def __init__(self, path: str):
        self.data = []
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def load(self) -> list[dict]:
        return self.data

    def statistics(self) -> dict:
        stat = {"doc": 0, "gen": 0, "kno": 0, "num": 0}
        for type_ in stat.keys():
            stat[type_] = sum([obj["type"] == type_ for obj in self.data])
        return stat
