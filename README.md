<h1 align="center">
    🍄 UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation
</h1>

<p align="center">
    <i>What does this repository include?</i><br>
    <b><a href="./eval/benchs/uhgeval/">UHGEval</a></b>: An unconstrained hallucination evaluation benchmark.<br>
    <b><a href="./eval/">Eval Suite</a></b>: A user-friendly evaluation framework for hallucination tasks.<br>
    Eval Suite supports other relevant benchmarks, such as <a href="https://github.com/OpenMOSS/HalluQA">HalluQA</a> and <a href="https://github.com/RUCAIBox/HaluEval">HaluEval</a>.
</p>

<p align="center">
    <a href="https://aclanthology.org/2024.acl-long.288/">
        <img alt="ACL Anthology Paper" src="https://img.shields.io/badge/ACL_Anthology-Paper-red.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgo8c3ZnCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgdmVyc2lvbj0iMS4wIgogICB3aWR0aD0iNjgiCiAgIGhlaWdodD0iNDYiCiAgIGlkPSJzdmcyIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNCIgLz4KICA8cGF0aAogICAgIGQ9Ik0gNDEuOTc3NTUzLC0yLjg0MjE3MDllLTAxNCBDIDQxLjk3NzU1MywxLjc2MTc4IDQxLjk3NzU1MywxLjQ0MjExIDQxLjk3NzU1MywzLjAxNTggTCA3LjQ4NjkwNTQsMy4wMTU4IEwgMCwzLjAxNTggTCAwLDEwLjUwMDc5IEwgMCwzOC40Nzg2NyBMIDAsNDYgTCA3LjQ4NjkwNTQsNDYgTCA0OS41MDA4MDIsNDYgTCA1Ni45ODc3MDgsNDYgTCA2OCw0NiBMIDY4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDEwLjUwMDc5IEwgNTYuOTg3NzA4LDMuMDE1OCBDIDU2Ljk4NzcwOCwxLjQ0MjExIDU2Ljk4NzcwOCwxLjc2MTc4IDU2Ljk4NzcwOCwtMi44NDIxNzA5ZS0wMTQgTCA0MS45Nzc1NTMsLTIuODQyMTcwOWUtMDE0IHogTSAxNS4wMTAxNTUsMTcuOTg1NzggTCA0MS45Nzc1NTMsMTcuOTg1NzggTCA0MS45Nzc1NTMsMzAuOTkzNjggTCAxNS4wMTAxNTUsMzAuOTkzNjggTCAxNS4wMTAxNTUsMTcuOTg1NzggeiAiCiAgICAgc3R5bGU9ImZpbGw6I2VkMWMyNDtmaWxsLW9wYWNpdHk6MTtmaWxsLXJ1bGU6ZXZlbm9kZDtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MTIuODk1NDExNDk7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLW1pdGVybGltaXQ6NDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLWRhc2hvZmZzZXQ6MDtzdHJva2Utb3BhY2l0eToxIgogICAgIGlkPSJyZWN0MjE3OCIgLz4KPC9zdmc+Cg==">
    </a>
    <a href="https://arxiv.org/abs/2311.15296">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-red.svg?logo=arxiv">
    </a>
    <a href="https://huggingface.co/datasets/Ki-Seki/UHGEvalDataset">
        <img alt="Hugging Face UHGEvalDataset" src="https://img.shields.io/badge/Hugging_Face-UHGEvalDataset-yellow?logo=huggingface">
    </a>
    <br>
    <a href="https://github.com/IAAR-Shanghai/UHGEval-dataset">
        <img alt="Static Badge" src="https://img.shields.io/badge/GitHub-Dataset_Creation-blue?logo=github">
    </a>
    <a href="https://opensource.org/license/apache-2-0/">
        <img alt="Apache 2.0 License" src="https://img.shields.io/badge/Apache_2.0-License-green.svg?logo=apache">
    </a>
</p>

## Quick Start

```bash
# Clone the repository
git clone https://github.com/IAAR-Shanghai/UHGEval.git
cd UHGEval

# Install dependencies
conda create -n uhg python=3.10
conda activate uhg
pip install -r requirements.txt

# Run evaluations with OpenAI Compatible API
python -m eval.cli eval openai \
    --model_name gpt-4o \
    --api_key your_api_key \
    --base_url https://api.openai.com/v1 \
    --evaluators ExampleQAEvaluator UHGSelectiveEvaluator

# Or run evaluations with Hugging Face Transformers
python -m eval.cli eval huggingface \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --apply_chat_template \
    --evaluators ExampleQAEvaluator UHGSelectiveEvaluator

# After evaluation, you can gather statistics of the evaluation results
python -m eval.cli stat

# List all available evaluators
python -m eval.cli list

# Get help
python -m eval.cli --help
```

> [!Tip]
> - Refer to [`demo.ipynb`](demo.ipynb) for more detailed examples.
> - Run `export HF_ENDPOINT=https://hf-mirror.com` to use the Chinese mirror if you cannot connect to Hugging Face.
> - SilliconFlow provides free API keys for many models, and you can apply for one at https://siliconflow.cn/pricing.

## UHGEval

UHGEval is a large-scale benchmark designed for evaluating hallucination in professional Chinese content generation. It builds on unconstrained text generation and hallucination collection, incorporating both automatic annotation and manual review.

**UHGEvalDataset.** UHGEval contains two dataset versions. The full version includes 5,141 data items, while a concise version with 1,000 items has been created for more efficient evaluation. Below is an example in UHGEvalDataset.

<details><summary><b>Example</b></summary>

```json
{
    "id": "num_000432",
    "headLine": "（社会）江苏首次评选消费者最喜爱的百种绿色食品",
    "broadcastDate": "2015-02-11 19:46:49",
    "type": "num",
    "newsBeginning": "  新华社南京2月11日电（记者李响）“民以食为天，食以安为先”。江苏11日发布“首届消费者最喜爱的绿色食品”评选结果，老山蜂蜜等100种食品获得消费者“最喜爱的绿色食品”称号。",
    "hallucinatedContinuation": "江苏是全国绿色食品生产最发达的省份之一。",
    "generatedBy": "InternLM_20B_Chat",
    "annotations": [
        "江苏<sep>合理",
        "全国<sep>合理",
        "绿色食品生产<sep>合理",
        "发达<sep>不合理，没有事实证明江苏是全国绿色食品生产发达的省份，但可以确定的是，江苏在绿色食品生产上有积极的实践和推动",
        "省份<sep>合理",
        "之一<sep>不合理，没有具体的事实证据表明江苏是全国绿色食品生产发达的省份之一"
    ],
    "realContinuation": "61家获奖生产企业共同签署诚信公约，共建绿色食品诚信联盟。",
    "newsRemainder": "61家获奖生产企业共同签署诚信公约，共建绿色食品诚信联盟。这是江苏保障食品安全、推动绿色食品生产的重要举措。\n  此次评选由江苏省绿色食品协会等部门主办，并得到江苏省农委、省委农工办、省工商局、省地税局、省信用办、省消协等单位大力支持。评选历时4个多月，经企业报名、组委会初筛、消费者投票等层层选拔，最终出炉的百强食品榜单由消费者亲自票选得出，网络、短信、报纸及现场投票共310多万份票数，充分说明了评选结果的含金量。\n  食品安全一直是社会关注的热点。此次评选过程中，组委会工作人员走街头、进超市，邀请媒体、消费者、专家深入产地开展绿色食品基地行，除了超市选购外，还搭建“诚信购微信商城”“中国移动MO生活绿色有机馆”等线上销售平台，开创江苏绿色食品“评展销”结合新局面。评选不仅宣传了江苏绿色品牌食品，更推动了省内绿色食品市场诚信体系的建立，为江苏绿色食品走向全国搭建了权威的平台。\n  江苏省农委副主任李俊超表示，绿色食品消费是当前社会重要的消费趋势。本次评选不仅为社会培育了食品安全诚信文化，也提高了消费者对食品质量和标识的甄别能力，实现了消费者和生产企业的“双赢”。\n  与会企业表示，能够入选“首届江苏消费者最喜爱的绿色食品”是消费者的信任和支持，他们将以此荣誉作为企业发展的新起点，严把食品质量关，推介放心安全的绿色品牌食品，促进产业稳定健康发展。（完）"
}
```

</details>

**Evaluation Methods.** UHGEval offers a variety of evaluation methods, including discriminative evaluation, generative evaluation, and selective evaluation.

| Evaluator                  | Metric                             | Description                                                                          |
| -------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------ |
| `UHGDiscKeywordEvaluator`  | Average Accuracy                   | Given a keyword, the LLM determines whether it contains hallucination.               |
| `UHGDiscSentenceEvaluator` | Average Accuracy                   | Given a sentence, the LLM determines whether it contains hallucination.              |
| `UHGGenerativeEvaluator`   | BLEU-4, ROUGE-L, kwPrec, BertScore | Given a continuation prompt, the LLM generates a continuation.                       |
| `UHGSelectiveEvaluator`    | Accuracy                           | Given hallucinated text and unhallucinated text, the LLM selects the realistic text. |

## Eval Suite

To facilitate evaluation, we have developed a user-friendly evaluation framework called Eval Suite. Currently, Eval Suite supports common hallucination evaluation benchmarks, allowing for comprehensive evaluation of the same LLM with just one command as shown in the [Quick Start](#quick-start) section.

| Benchmark | Evaluator                                                                                                      | More Information                               |
| --------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| ExampleQA | `ExampleQAEvaluator`                                                                                           | [eval/benchs/exampleqa](eval/benchs/exampleqa) |
| HalluQA   | `HalluQAMCEvaluator`                                                                                           | [eval/benchs/halluqa](eval/benchs/halluqa)     |
| HaluEval  | `HaluEvalDialogEvaluator`<br>`HaluEvalQAEvaluator`<br>`HaluEvalSummaEvaluator`                                 | [eval/benchs/halueval](eval/benchs/halueval)   |
| UHGEval   | `UHGDiscKeywordEvaluator`<br>`UHGDiscSentenceEvaluator`<br>`UHGGenerativeEvaluator`<br>`UHGSelectiveEvaluator` | [eval/benchs/uhgeval](eval/benchs/uhgeval)     |

## Learn More

- [Eval Suite architecture](docs/architecture.md)
- [Add new benchmarks or model loaders](docs/add-bench-or-model.md)
- [Some experiment results](docs/experiments.md)

## Citation

```bibtex
@article{liang2023uhgeval,
  title={Uhgeval: Benchmarking the hallucination of chinese large language models via unconstrained generation},
  author={Liang, Xun and Song, Shichao and Niu, Simin and Li, Zhiyu and Xiong, Feiyu and Tang, Bo and Wy, Zhaohui and He, Dawei and Cheng, Peng and Wang, Zhonghao and others},
  journal={arXiv preprint arXiv:2311.15296},
  year={2023}
}
```

## TODOs

<details><summary>Click me to show all TODOs</summary>

- [ ] feat: vLLM offline inference benchmarking
- [ ] build: packaging
- [ ] feat(benchs): add TruthfulQA benchmark
- [ ] docs: update citation with DOI
- [ ] other: promotion

</details>
