# @Author : YeZhaohui Wang
# @Email  : wyzh0912@126.com

import unittest

from uhgeval.dataset.truthfulqa import TruthfunQAGeneration


class TestTruthfulQAGen(unittest.TestCase):

    def setUp(self):
        # Path to the sample data file
        self.dataset_path = "data/TruthfulQA/TruthfulQA.csv"

    def test_init(self):
        dataset = TruthfunQAGeneration(self.dataset_path)
        self.assertEqual(len(dataset), 817)
        self.assertIsInstance(dataset.data, list)

    def test_shuffle(self):
        dataset = TruthfunQAGeneration(
            self.dataset_path, shuffle=True, seed=42)
        self.assertNotEqual(
            dataset.data, TruthfunQAGeneration(
                self.dataset_path).data)
        self.assertEqual(sorted(dataset.data, key=lambda x: x["id"]),
                         sorted(TruthfunQAGeneration(self.dataset_path).data, key=lambda x: x["id"]))

    def test_getitem(self):
        dataset = TruthfunQAGeneration(self.dataset_path)
        index = 2
        self.assertEqual(
            dataset[index], TruthfunQAGeneration(
                self.dataset_path).data[index])

    def test_load(self):
        dataset = TruthfunQAGeneration(self.dataset_path)
        loaded_data = dataset.load()
        self.assertEqual(
            loaded_data, TruthfunQAGeneration(
                self.dataset_path).data)


if __name__ == '__main__':
    unittest.main()
