# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import json
import os
import random

from uhgeval.dataset.base import BaseDataset


class HaluEvalQA(BaseDataset):
    def __init__(self, path: str, shuffle: bool = False, seed: int = 22):
        self.data = []
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                for id, line in enumerate(f):
                    self.data.append({'id': id, **json.loads(line)})
        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
            return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]


class HaluEvalDialogue(BaseDataset):
    def __init__(self, path: str, shuffle: bool = False, seed: int = 22):
        self.data = []
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                for id, line in enumerate(f):
                    self.data.append({'id': id, **json.loads(line)})
        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
            return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]


class HaluEvalSummarization(BaseDataset):
    def __init__(self, path: str, shuffle: bool = False, seed: int = 22):
        self.data = []
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                for id, line in enumerate(f):
                    self.data.append({'id': id, **json.loads(line)})
        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
            return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]
