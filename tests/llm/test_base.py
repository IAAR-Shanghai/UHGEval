# @Author : Shichao Song
# @Email  : song.shichao@outlook.com


import unittest
from unittest.mock import patch


from uhgeval.llm.base import BaseLLM


class ConcreteLLM(BaseLLM):
    def request(self, query: str) -> str:
        return 'Mocked Response'


class TestBaseLLM(unittest.TestCase):

    def setUp(self):
        # Initialize an instance of BaseLLM's child class for testing
        self.llm_instance = ConcreteLLM()

    def test_init(self):
        self.assertEqual(self.llm_instance.params['model_name'], 'ConcreteLLM')
        self.assertEqual(self.llm_instance.params['temperature'], 1.0)
        self.assertEqual(self.llm_instance.params['max_new_tokens'], 1024)
        self.assertEqual(self.llm_instance.params['top_p'], 0.9)
        self.assertEqual(self.llm_instance.params['top_k'], 5)

    def test_update_params(self):
        new_params = {'model_name': 'NewModel', 'temperature': 0.8}
        updated_instance = self.llm_instance.update_params(**new_params)
        self.assertEqual(updated_instance.params['model_name'], 'NewModel')
        self.assertEqual(updated_instance.params['temperature'], 0.8)

    @patch.object(ConcreteLLM, 'request', return_value='mocked_response')
    def test_safe_request(self, mock_request):
        response = self.llm_instance.safe_request('query')
        mock_request.assert_called_once_with('query')
        self.assertEqual(response, 'mocked_response')

    def test_continue_writing(self):
        obj = {'headLine': 'Title', 'broadcastDate': '2023-01-01', 'newsBeginning': 'Lorem ipsum'}
        with patch.object(ConcreteLLM, 'safe_request', return_value='<response>Mocked Response</response>'):
            result = self.llm_instance.continue_writing(obj)
        self.assertEqual(result, 'Mocked Response')

    @patch.object(ConcreteLLM, 'safe_request', return_value='<keywords>keyword1\nkeyword2</keywords>')
    def test_extract_kws(self, mock_safe_request):
        sentence = 'This is a test sentence with keyword1 and keyword2.'
        keywords = self.llm_instance.extract_kws(sentence)
        mock_safe_request.assert_called_once()
        self.assertEqual(keywords, ['keyword1', 'keyword2'])

    @patch.object(ConcreteLLM, 'safe_request', return_value='不符合现实。原因。')
    def test_is_kw_hallucinated(self, mock_safe_request):
        kw = 'keyword'
        obj = {'headLine': 'Title', 'broadcastDate': '2023-01-01', 'newsBeginning': 'Lorem ipsum', 'hallucinatedContinuation': 'Hallucinated continuation'}
        result, reason = self.llm_instance.is_kw_hallucinated(kw, obj, with_reason=True)
        mock_safe_request.assert_called_once()
        self.assertIn(result, {0, 1, -1})
        self.assertIsInstance(reason, str)

    @patch.object(ConcreteLLM, 'safe_request', return_value='B。原因。')
    def test_compare_two_continuation(self, mock_safe_request):
        contn1 = 'Continuation 1'
        contn2 = 'Continuation 2'
        obj = {'headLine': 'Title', 'broadcastDate': '2023-01-01', 'newsBeginning': 'Lorem ipsum'}
        result = self.llm_instance.compare_two_continuation(contn1, contn2, obj)
        mock_safe_request.assert_called_once()
        self.assertIn(result, {1, 2, -1})

    @patch.object(ConcreteLLM, 'safe_request', return_value='续写不符合现实。原因。')
    def test_is_continuation_hallucinated(self, mock_safe_request):
        contn = 'Continuation'
        obj = {'headLine': 'Title', 'broadcastDate': '2023-01-01', 'newsBeginning': 'Lorem ipsum'}
        result, reason = self.llm_instance.is_continuation_hallucinated(contn, obj, with_reason=True)
        mock_safe_request.assert_called_once()
        self.assertIn(result, {0, 1, -1})
        self.assertIsInstance(reason, str)

    def test_read_prompt_template(self):
        with patch('builtins.open', return_value=open('uhgeval/prompts/continue_writing.txt', 'r')):
            template = self.llm_instance._read_prompt_template('continue_writing.txt')
        self.assertTrue(template.startswith('你是一名新华社新闻工作者。我希望你能辅助我完成一篇新闻的撰写。'))


if __name__ == '__main__':
    unittest.main()
