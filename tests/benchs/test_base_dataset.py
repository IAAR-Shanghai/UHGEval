import unittest

from eval_suite.benchs.base_dataset import DummyDataset


class TestDummyDataset(unittest.TestCase):
    def setUp(self):
        """Set up the DummyDataset instance for testing."""
        self.dataset = DummyDataset()

    def test_load(self):
        """Test that the `load` method returns the correct dataset."""
        expected_data = [
            {"id": "1", "type": "math", "q": "1+1=", "a": "2"},
            {"id": "2", "type": "geo", "q": "France capital is?", "a": "Paris"},
        ]
        self.assertEqual(self.dataset.load(), expected_data)

    def test_to_batched(self):
        """Test the `to_batched` method for correct batching behavior."""
        # Test with 2 batches
        expected_batches = [
            [{"id": "1", "type": "math", "q": "1+1=", "a": "2"}],
            [{"id": "2", "type": "geo", "q": "France capital is?", "a": "Paris"}],
        ]
        self.assertEqual(self.dataset.to_batched(2), expected_batches)

        # Test with 1 batch (all data in one batch)
        expected_batches = [
            [
                {"id": "1", "type": "math", "q": "1+1=", "a": "2"},
                {"id": "2", "type": "geo", "q": "France capital is?", "a": "Paris"},
            ]
        ]
        self.assertEqual(self.dataset.to_batched(1), expected_batches)

        # Test with more batches than data points (each batch should have 1 data point)
        expected_batches = [
            [{"id": "1", "type": "math", "q": "1+1=", "a": "2"}],
            [{"id": "2", "type": "geo", "q": "France capital is?", "a": "Paris"}],
        ]
        self.assertEqual(self.dataset.to_batched(3), expected_batches)

    def test_to_batched_empty_dataset(self):
        """Test the `to_batched` method with an empty dataset."""
        self.dataset.data = []  # Set dataset to empty
        expected_batches = [[]]  # Should return a single empty batch
        self.assertEqual(self.dataset.to_batched(1), expected_batches)

    def test_to_batched_with_remainder(self):
        """Test the `to_batched` method with batches that do not evenly divide the dataset."""
        # Add more data to create an uneven number of items
        self.dataset.data.append(
            {"id": "3", "type": "math", "q": "What is 2+2?", "a": "4"}
        )
        expected_batches = [
            [
                {"id": "1", "type": "math", "q": "1+1=", "a": "2"},
                {"id": "2", "type": "geo", "q": "France capital is?", "a": "Paris"},
            ],
            [{"id": "3", "type": "math", "q": "What is 2+2?", "a": "4"}],
        ]
        self.assertEqual(self.dataset.to_batched(2), expected_batches)


if __name__ == "__main__":
    unittest.main()
