# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import unittest


from uhgeval.metric.common import (
    bleu4_score,
    rougeL_score,
    kw_precision,
    bert_score,
    classifications
)


class TestEvaluateFunctions(unittest.TestCase):

    def test_bleu4_score(self):
        continuation = "This is a test sentence."
        reference = "This is a reference sentence."
        score = bleu4_score(continuation, reference)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_rougeL_score(self):
        continuation = "This is a test sentence."
        reference = "This is a reference sentence."
        score = rougeL_score(continuation, reference)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_kw_precision(self):
        continuation = "This is a test sentence with keywords: keyword1 keyword2."
        reference = "This is a reference sentence with keyword1."
        kw_extracter = lambda text: text.split()[-2:]
        precision, appeared_kws, all_kws = kw_precision(continuation, reference, kw_extracter, with_kw_list=True)
        self.assertIsInstance(precision, float)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertIsInstance(appeared_kws, list)
        self.assertIsInstance(all_kws, list)

    def test_bert_score(self):
        continuation = "This is a test sentence."
        reference = "This is a reference sentence."
        score = bert_score(continuation, reference)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_classifications(self):
        predictions = [True, False, True, True, False]
        references = [True, True, False, True, True]
        accuracy, precision, recall, f1 = classifications(predictions, references)
        self.assertEqual(accuracy, 0.4)
        self.assertAlmostEqual(precision, 0.6666666666666666)
        self.assertEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5714285714285715)


if __name__ == '__main__':
    unittest.main()
