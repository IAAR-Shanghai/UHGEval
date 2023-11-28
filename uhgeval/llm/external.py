from uhgeval.llm.base import BaseLLM

from transformers import AutoTokenizer, AutoModel


class ChatGLM3_6B_Chat(BaseLLM):
    def request(self, query: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
        model = model.eval()
        response, _ = model.chat(tokenizer, query, history=[])
        return response
