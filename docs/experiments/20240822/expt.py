from eval_suite.benchs import (
    UHGDiscKeywordEvaluator,
    UHGDiscSentenceEvaluator,
    UHGGenerativeEvaluator,
    UHGSelectiveEvaluator,
)
from eval_suite.llms import OpenAIAPI

glm = OpenAIAPI(
    model_name="THUDM/glm-4-9b-chat",
    api_key="your_api_key",
    base_url="https://api.siliconflow.cn/v1",
)

qwen = OpenAIAPI(
    model_name="Qwen/Qwen1.5-7B-Chat",
    api_key="your_api_key",
    base_url="https://api.siliconflow.cn/v1",
)

llama = OpenAIAPI(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key="your_api_key",
    base_url="https://api.siliconflow.cn/v1",
)

for model in (glm, qwen, llama):
    for use_full in (True, False):
        for evaluator in [
            UHGSelectiveEvaluator,
            UHGGenerativeEvaluator,
            UHGDiscSentenceEvaluator,
            UHGDiscKeywordEvaluator,
        ]:
            evaluator(model=model, use_full=use_full).evaluate()
