import unittest

from eval.metrics import bert_score, bleu_4, keyword_precision, rouge_l


class TestEvaluationFunctions(unittest.TestCase):

    def test_bleu_4(self):
        predication = "我非常高兴见到你。"
        reference = "我非常高兴见到你。"
        self.assertEqual(bleu_4(predication, reference), 1.0)

    def test_rouge_l(self):
        predication = "这是一个测试"
        reference = "这是一个测试"
        self.assertEqual(rouge_l(predication, reference), 1.0)

    def test_keyword_precision(self):
        keywords = ["测试", "一个", "例子"]
        reference = "这是一个测试"

        self.assertEqual(keyword_precision(keywords, reference), 2 / 3)
        self.assertEqual(keyword_precision([], reference), 0)
        self.assertEqual(keyword_precision(["不存在", "错误"], reference), 0.0)

    def test_bert_score(self):
        continuation = "这是一个测试"
        reference = "这是一个测试"
        self.assertEqual(bert_score(continuation, reference), 1.0)


if __name__ == "__main__":
    unittest.main()
