import math
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base class for dataset loading."""

    @abstractmethod
    def __init__(self):
        """Initialize the dataset."""
        ...

    @abstractmethod
    def load(self) -> list[dict]:
        """Return the full dataset.

        For example:
            ```
            return [
                {"id": "1", "type": "math", "q": "1+1=", "a": "2"},
                {"id": "2", "type": "geo", "q": "France capital is?", "a": "Paris"},
            ]
            ```

        Note:
            Each item within the dataset must be a dictionary and must include:
            * `id`: A unique identifier as a string.
            * `type`: (Optional, but recommended) A string indicating the type of data, e.g., 'math', 'geo'.
        """
        ...

    def to_batched(self, num_batches: int) -> list[list[dict]]:
        """Return the dataset divided into batches.

        Args:
            num_batches (int): The number of batches to divide the dataset into.

        Returns:
            list[list[dict]]: A list of batches, each containing a list of data points.

        Note:
            The last batch may have fewer data points than the other batches.
        """
        if num_batches <= 0:
            raise ValueError("num_batches must be greater than 0.")

        dataset = self.load()
        if not dataset:
            return [[]]

        batch_size = math.ceil(len(dataset) / num_batches)
        batches = [
            dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)
        ]
        return batches


class DummyDataset(BaseDataset):
    def __init__(self):
        self.data = [
            {"id": "1", "type": "math", "q": "1+1=", "a": "2"},
            {"id": "2", "type": "geo", "q": "France capital is?", "a": "Paris"},
        ]

    def load(self) -> list[dict]:
        return self.data
