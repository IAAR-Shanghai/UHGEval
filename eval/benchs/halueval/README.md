# HaluEval

## Information

- **Paper**: HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models
- **Institution**:
  - Renmin University of China
  - Université de Montréal
  - Beijing Key Laboratory of Big Data Management and Analysis Methods
- **ACL Anthology**: https://aclanthology.org/2023.emnlp-main.397/
- **arXiv**: https://arxiv.org/abs/2305.11747
- **GitHub**: https://github.com/RUCAIBox/HaluEval

## Evaluators

| Evaluator                 | Metric   | Description                              |
| ------------------------- | -------- | ---------------------------------------- |
| `HaluEvalDialogEvaluator` | Accuracy | Hallucination detection in dialogue      |
| `HaluEvalQAEvaluator`     | Accuracy | Hallucination detection in QA            |
| `HaluEvalSummaEvaluator`  | Accuracy | Hallucination detection in summarization |

## Citation

```bibtex
@inproceedings{li-etal-2023-halueval,
    title = "{H}alu{E}val: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models",
    author = "Li, Junyi  and
      Cheng, Xiaoxue  and
      Zhao, Xin  and
      Nie, Jian-Yun  and
      Wen, Ji-Rong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.397",
    doi = "10.18653/v1/2023.emnlp-main.397",
    pages = "6449--6464",
    abstract = "Large language models (LLMs), such as ChatGPT, are prone to generate hallucinations, i.e., content that conflicts with the source or cannot be verified by the factual knowledge. To understand what types of content and to which extent LLMs are apt to hallucinate, we introduce the Hallucination Evaluation for Large Language Models (HaluEval) benchmark, a large collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognizing hallucination. To generate these samples, we propose a ChatGPT-based two-step framework, i.e., sampling-then-filtering. Besides, we also hire some human labelers to annotate the hallucinations in ChatGPT responses. The empirical results suggest that ChatGPT is likely to generate hallucinated content in specific topics by fabricating unverifiable information (i.e., about 19.5{\%} user queries). Moreover, existing LLMs face great challenges in recognizing the hallucinations in texts. While, our experiments also prove that the hallucination recognition can be improved by providing external knowledge or adding reasoning steps.",
}
```
