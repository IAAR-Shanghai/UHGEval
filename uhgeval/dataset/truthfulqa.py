# @Author : YeZhaohui Wang
# @Email  : wyzh0912@126.com

import csv
import json
import os
import random

from uhgeval.dataset.base import BaseDataset


class TruthfunQAGeneration(BaseDataset):
    def __init__(self, path: str, shuffle: bool = False, seed: int = 22):
        self.data = []
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8-sig') as file:
                csv_reader = csv.DictReader(file)
                id = 1
                for row in csv_reader:
                    row['id'] = id
                    id += 1
                    self.data.append(row)
        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]


class TruthfunQAMC1(BaseDataset):
    def __init__(self, path: str, shuffle: bool = False, seed: int = 22):
        self.data = []
        id = 1
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                self.data = json.load(f)
            for row in self.data:
                row['id'] = id
                id += 1

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]


class TruthfunQAMC2(BaseDataset):
    def __init__(self, path: str, shuffle: bool = False, seed: int = 22):
        self.data = []
        id = 1
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                self.data = json.load(f)
            for row in self.data:
                row['id'] = id
                id += 1

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]
