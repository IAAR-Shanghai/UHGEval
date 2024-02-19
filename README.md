<h1 align="center">
    🍄 UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation
</h1>
<p align="center">
    Safety: Ensuring the security of experimental data is of utmost importance.<br>
    Flexibility: Easily expandable, with all modules replaceable.
</p>
<p align="center">
    <a href="https://opensource.org/license/apache-2-0/">
        <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-yellow.svg">
    </a>
    <a href="https://arxiv.org/abs/2311.15296">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
    </a>
    <a href="./data/Xinhua/XinhuaHallucinations.json">
        <img alt="Static Badge" src="https://img.shields.io/badge/Dataset-XinhuaHallucination-blue">
    </a>
    <a href="https://github.com/IAAR-Shanghai/UHGEval-dataset">
        <img alt="Static Badge" src="https://img.shields.io/badge/GitHub-Dataset_Creation_Pipeline-green">
    </a>
</p>

## What's New 🆕

- **2024.1.14**: Added local model loading for evaluation.
- **2024.1.13**: Introduced experiment initialization via `config.yaml`.
- **2024.1.12**: Supported [TruthfulQA](https://github.com/sylinrl/TruthfulQA) with generative and Multi-Choices evaluation.

<details><summary>Click me to show all TODOs</summary>

- [ ] refactor: interface design
- [ ] contribution: OpenCompass

</details>

## Introduction

UHGEval is a comprehensive framework designed for evaluating the hallucination phenomena in Chinese large language models (LLMs) through unconstrained text generation. Its architecture offers flexibility and extensibility, allowing for easy integration of new datasets, models, and evaluation metrics.

<p align="center"><img src="./assets/eval_framework.png" alt="" width="80%"></p>

Additionally, we've made the dataset creation process transparent and accessible through our open-source pipeline, [UHGEval-dataset](https://github.com/IAAR-Shanghai/UHGEval-dataset). This enables researchers to craft customized datasets. UHGEval supports seamless integration of these datasets, facilitating comprehensive evaluations. A prime example is our incorporation of the [TruthfulQA](https://github.com/sylinrl/TruthfulQA) dataset, showcasing the framework's capability to adapt to diverse evaluation needs.

## Project Structure

```bash
.
├── .github
├── .gitignore
├── CITATION.bib
├── LICENSE
├── README.md
├── README.zh_CN.md
├── archived_experiments    # Experiment results no longer in active use
├── assets                  # Static files like images used in documentation
├── config.yaml             # Configuration file for initializing experiments
├── data                    # Datasets (e.g., XinhuaHallucinations)
├── demo.py                 # A demonstration script showcasing the project's capabilities
├── logs                    # Errors, warnings, information, etc.
├── output                  # Stores the results of evaluations
├── requirements.txt
├── run_uhgeval.py          # Script to run the evaluation
├── run_uhgeval_future.py   # Script to run the evaluation
├── statistics              # Holds summary results of multiple experiments
├── tests                   # Contains unit testing scripts
└── uhgeval                 # Source code for the project
    ├── .cache              # Cache folder for storing temporary data or scripts
    ├── configs             # Scripts for initializing model loading parameters
    ├── core                # Core scripts for data analysis and experiment orchestration
    ├── dataset             # Scripts for dataset loading
    ├── evaluator           # Scripts for performing evaluation tasks
    ├── llm                 # Scripts for loading and interacting with LLMs
    ├── metric              # Defines evaluation metrics
    └── prompts             # Prompt Engineering
```

## Quick Start

Get started quickly with a 20-line demo program.

* UHGEval requires Python>=3.10.0
* `pip install -r requirements.txt`
* Take `uhgeval/configs/example_config.py` as an example, create `uhgeval/configs/real_config.py` and add configs of models you want to evaluate to this script.
* Run `demo.py`

## Advanced Usage

Utilize `run_uhgeval.py` or `run_uhgeval_future.py` for a comprehensive understanding of this project. The former is currently a **provisional piece** of code slated for removal in the future; whereas the latter is **command-line executable** code intended for future use. Both scripts will perform a complete evaluation process using the GPT model. 

## Results for Experiment-20231117

<p align="center"><img src="./assets/discri_and_sel.png" alt=""></p>

<p align="center"><img src="./assets/gen.png" alt=""></p>

<p align="center"><img src="./assets/by_type.png" alt="" width="60%"></p>

The original experimental results are in [./archived_experiments/20231117](./archived_experiments/20231117).

## Customization Guideline

### Adding New Datasets
- **Place Dataset Files**: Add your dataset files under `data/dataset_name/`, e.g., `data/MyDataset/A.json`.
- **Implement Loader Script**: Create a script for loading your dataset at `uhgeval/dataset/loadMyDataset.py`.

### Integrating New Models
- **Identify Model Type**: Determine how your model is loaded (API, local, etc.).
- **Add Model Loader**:
  - For API models: `uhgeval/llm/apiModelLoader.py`.
  - Adapt the path based on the model type.

### Creating Custom Metrics
- **Implement Metric**: Add your metric implementation in `uhgeval/metric/`.

### Developing New Evaluators
- **Implement Evaluator**: Craft your evaluator script, ensuring it aligns with the structure of `base.py`, and place it at `uhgeval/evaluator/yourEvaluator.py`.
- **Model Interaction Method**: Introduce a method for model interaction in `uhgeval/llm/base.py`.
- **Prepare Prompt Files**: Create prompt files for your evaluator at `uhgeval/prompts/yourPrompts.txt`.

## Contributions

Although we have conducted thorough automatic annotation and manual verification, there may still be errors or imperfections in our [XinhuaHallucinations](./data/Xinhua/XinhuaHallucinations.json) dataset with over 5000 data points. We encourage you to raise issues or submit pull requests to assist us in improving the consistency of the dataset. You may also receive corresponding recognition and rewards for your contributions.

You can also contribute to the project by adding new datasets, integrating new models, creating custom metrics, or developing new evaluators. We welcome all forms of contributions.

> [!Note]
> Remember to read the [Contribution Guidelines](./.github/CONTRIBUTING.md) before creating a pull request!

## Citation

```BibTeX
@article{UHGEval,
    title={UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation},
    author={Xun Liang and Shichao Song and Simin Niu and Zhiyu Li and Feiyu Xiong and Bo Tang and Zhaohui Wy and Dawei He and Peng Cheng and Zhonghao Wang and Haiying Deng},
    journal={arXiv preprint arXiv:2311.15296},
    year={2023},
}
```
