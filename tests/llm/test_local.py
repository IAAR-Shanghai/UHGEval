# @Author : YeZhaohui Wang
# @Email  : wyzh0912@126.com

"""Unit tests for the uhgeval.llm.base module.

This module contains unittests for the llm deployed locally.

Note:
    These tests perform real requests to local models. Please ensure the integrity of the model files
"""


import unittest

from uhgeval.llm.local import (Aquila_34B_Chat, Baichuan2_13B_Chat, ChatGLM3_6B_Chat,
                               InternLM_20B_Chat, Qwen_14B_Chat)


class BaseChatTest(unittest.TestCase):
    def _test_request(self):
        query = "How are you?"
        response = self.model.request(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def _test_continue_writing(self):
        obj = {"headLine": "Story", "broadcastDate": "2023-11-15",
               "newsBeginning": "Once upon a time, there is a"}
        result = self.model.continue_writing(obj)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestAquila34BChatlocal(unittest.TestCase):
    def _test_request(self):
        query = "How are you?"
        response = self.model.request(query)
        self.assertIsInstance(response, str)
        self.assertGreaterEqual(len(response), 0)

    def setUp(self):
        self.model = Aquila_34B_Chat()

    def test_request(self):
        self._test_request()


class TestBaichuan213BChatlocal(BaseChatTest):
    def setUp(self):
        self.model = Baichuan2_13B_Chat()

    def test_request(self):
        self._test_request()


class TestChatGLM36BChatlocal(BaseChatTest):
    def setUp(self):
        self.model = ChatGLM3_6B_Chat()

    def test_request(self):
        self._test_request()


class TestQwen14BChatlocal(BaseChatTest):
    def setUp(self):
        self.model = Qwen_14B_Chat()

    def test_request(self):
        self._test_request()


class TestInternLM20BChat(BaseChatTest):
    def setUp(self):
        self.model = InternLM_20B_Chat()

    def test_request(self):
        self._test_request()


if __name__ == '__main__':
    unittest.main()
