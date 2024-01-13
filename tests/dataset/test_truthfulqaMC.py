# @Author : YeZhaohui Wang
# @Email  : wyzh0912@126.com


import unittest

from uhgeval.dataset.truthfulqa import TruthfunQAMC1


class TestTruthfulQAMC(unittest.TestCase):

    def setUp(self):
        # Path to the sample data file
        self.dataset_path = "data/TruthfulQA/mc_task.json"

    def test_init(self):
        dataset = TruthfunQAMC1(self.dataset_path)
        self.assertEqual(len(dataset), 817)
        self.assertIsInstance(dataset.data, list)

    def test_shuffle(self):
        dataset = TruthfunQAMC1(self.dataset_path, shuffle=True, seed=42)
        self.assertNotEqual(
            dataset.data, TruthfunQAMC1(
                self.dataset_path).data)
        self.assertEqual(sorted(dataset.data, key=lambda x: x["id"]),
                         sorted(TruthfunQAMC1(self.dataset_path).data, key=lambda x: x["id"]))

    def test_getitem(self):
        dataset = TruthfunQAMC1(self.dataset_path)
        index = 2
        self.assertEqual(
            dataset[index], TruthfunQAMC1(
                self.dataset_path).data[index])

    def test_load(self):
        dataset = TruthfunQAMC1(self.dataset_path)
        loaded_data = dataset.load()
        self.assertEqual(loaded_data, TruthfunQAMC1(self.dataset_path).data)


if __name__ == '__main__':
    unittest.main()
