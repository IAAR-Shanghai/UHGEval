# @Author : Shichao Song
# @Email  : song.shichao@outlook.com

"""Unit tests for the uhgeval.llm.remote module.

This module contains unittests for the llm deployed remotely.

Note:
    These tests perform real requests to external APIs. Be cautious of network availability,
    API rate limits, and potential costs associated with making real requests during testing.
"""


import unittest

from uhgeval.llm.remote import (
    Aquila_34B_Chat,
    Baichuan2_13B_Chat,
    Baichuan2_53B_Chat,
    ChatGLM2_6B_Chat,
    GPT_transit,
    InternLM_20B_Chat,
    Qwen_14B_Chat,
    Xinyu_7B_Chat,
    Xinyu_70B_Chat,
)


class BaseChatTest(unittest.TestCase):
    def _test_request(self):
        query = "How are you?"
        response = self.model.request(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def _test_continue_writing(self):
        obj = {"headLine": "Story", "broadcastDate": "2023-11-15", "newsBeginning": "Once upon a time, there is a"}
        result = self.model.continue_writing(obj)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestAquila34BChat(BaseChatTest):
    def setUp(self):
        self.model = Aquila_34B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


class TestBaichuan213BChat(BaseChatTest):
    def setUp(self):
        self.model = Baichuan2_13B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


class TestBaichuan253BChat(BaseChatTest):
    def setUp(self):
        self.model = Baichuan2_53B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


class TestChatGLM26BChat(BaseChatTest):
    def setUp(self):
        self.model = ChatGLM2_6B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


class TestGPTTransit(BaseChatTest):
    def setUp(self):
        self.gpt35 = GPT_transit(model_name='gpt-3.5-turbo', temperature=0.1)
        self.gpt4_0613 = GPT_transit(model_name='gpt-4-0613', temperature=0.1)
        self.gpt4_1106 = GPT_transit(model_name='gpt-4-1106-preview', temperature=0.1)

    def _test_request(self, model):
        query = "How are you?"
        response = model.request(query)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_request(self):
        for model in [self.gpt35, self.gpt4_0613, self.gpt4_1106]:
            with self.subTest(model=model):
                self._test_request(model)

    def _test_continue_writing(self, model):
        obj = {"headLine": "Story", "broadcastDate": "2023-11-15", "newsBeginning": "Once upon a time, there is a"}
        result = model.continue_writing(obj)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_continue_writing(self):
        for model in [self.gpt35, self.gpt4_0613, self.gpt4_1106]:
            with self.subTest(model=model):
                self._test_continue_writing(model)


class TestInternLM20BChat(BaseChatTest):
    def setUp(self):
        self.model = InternLM_20B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


class TestQwen14BChat(BaseChatTest):
    def setUp(self):
        self.model = Qwen_14B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


class TestXinyu7BChat(BaseChatTest):
    def setUp(self):
        self.model = Xinyu_7B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


class TestXinyu70BChat(BaseChatTest):
    def setUp(self):
        self.model = Xinyu_70B_Chat(temperature=0.1)

    def test_request(self):
        self._test_request()

    def test_continue_writing(self):
        self._test_continue_writing()


if __name__ == '__main__':
    unittest.main()
