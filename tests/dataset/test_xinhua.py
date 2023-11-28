# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import unittest

from uhgeval.dataset.xinhua import XinhuaHallucinations


class TestXinhuaHallucinations(unittest.TestCase):

    def setUp(self):
        # Path to the sample data file
        self.dataset_path = "data/Xinhua/XinhuaHallucinations.json"

    def test_init(self):
        dataset = XinhuaHallucinations(self.dataset_path)
        self.assertEqual(len(dataset), 5141)
        self.assertIsInstance(dataset.data, list)

    def test_shuffle(self):
        dataset = XinhuaHallucinations(self.dataset_path, shuffle=True, seed=42)
        self.assertNotEqual(dataset.data, XinhuaHallucinations(self.dataset_path).data)
        self.assertEqual(sorted(dataset.data, key=lambda x: x["id"]),
                         sorted(XinhuaHallucinations(self.dataset_path).data, key=lambda x: x["id"]))

    def test_getitem(self):
        dataset = XinhuaHallucinations(self.dataset_path)
        index = 2
        self.assertEqual(dataset[index], XinhuaHallucinations(self.dataset_path).data[index])

    def test_load(self):
        dataset = XinhuaHallucinations(self.dataset_path)
        loaded_data = dataset.load()
        self.assertEqual(loaded_data, XinhuaHallucinations(self.dataset_path).data)

    def test_statistics(self):
        dataset = XinhuaHallucinations(self.dataset_path)
        stats = dataset.statistics()
        types_in_data = set(obj['type'] for obj in XinhuaHallucinations(self.dataset_path).data)
        expected_stats = {type_: sum(obj['type'] == type_ for obj in XinhuaHallucinations(self.dataset_path).data)
                         for type_ in types_in_data}
        self.assertEqual(stats, expected_stats)


if __name__ == '__main__':
    unittest.main()
