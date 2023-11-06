import json
import os
import random
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __init__(self, path):
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        ...

    @abstractmethod
    def load(self) -> list[dict]:
        ...


class XinhuaHallucinations(Dataset):
    def __init__(self, path: str, shuffle: bool = False, seed: int = 22):
        self.data = []
        if os.path.isfile(path):
            with open(path) as f:
                self.data = json.load(f)
        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
            return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]

    def statistics(self) -> dict:
        stat = {'doc': 0, 'gen': 0, 'kno': 0, 'num': 0}
        for type_ in stat.keys():
            stat[type_] = sum([obj['type']==type_ for obj in self.data])
        return stat
