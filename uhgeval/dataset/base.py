# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """
    Base class for datasets ensuring each data item is structured as a dictionary.
    
    Each item within the dataset must include:
    - `id`: A unique identifier as a string.
    - `type`: (Optional, but recommended) A string indicating the type of data, e.g., 'math', 'geo'.
    
    A well-formed dataset example:
    
    dataset = [
        {'id': '001', 'type': 'math', 'question': '1+1=', 'answer': '2'},
        {'id': '002', 'type': 'geo', 'question': 'What is the capital of France?', 'answer': 'Paris'}
    ]
    """

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
