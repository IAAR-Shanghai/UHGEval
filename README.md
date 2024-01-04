[English](./README.md) | [中文简体](./README.zh_CN.md)

<a href="https://opensource.org/license/apache-2-0/">
    <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-yellow.svg">
</a>
<a href="https://arxiv.org/abs/2311.15296">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
</a>

# Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
  * [uhgeval](#uhgeval)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Customization Guideline](#customization-guideline)
- [Results for Experiment-20231117](#results-for-experiment-20231117)
- [Contributions](#contributions)
  * [TODOs](#todos)
  * [CITATION](#citation)

# 🍄 Introduction

Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation

* Safety: Ensuring the security of experimental data is of utmost importance.
* Flexibility: Easily expandable, with all modules replaceable.

<p align="center"><img src="./assets/eval_framework.png" alt="" width="80%"></p>

# Project Structure

The project contains multiple folders and multiple Python script files. Here are the introductions of them.
* **data**: This folder comprises the datasets used for evaluation. Currently, it only includes the **XinhuaHallucinations** dataset. You can add your **own dataset** here.
* **logs**: This folder is used to store logs, warnings, errors, information, etc., such as token costs, network connection problems, incorrect responses, etc.
* **output**: This folder is used to store the results of the evaluation.
* **statistics**: This folder is used to store summary results of **multiple experiments**. 
* **tests**: This folder includes scripts to run **unit testing**. 
* **uhgeval**: This folder contains the core components of this project. We provide a more detailed introduction below.
## uhgeval
The files contained in this folder ensure that the project runs properly. Here we will explain their respective roles.
* **configs**: This folder comprises scripts used to **initialize** the **loading parameters** of the model. There are lots of strings in this script. The string can be an API, a URL, or a local path, depending on your way of loading.
* **dataset**: This folder contains scripts used to **load the dataset**. Currently, it only includes the script for loading the XinhuaHallucinations dataset. You can add your own loading script here. 
* **evaluator**: This folder includes scripts used to **perform the evaluation tasks**. Here we provide scripts for performing discriminative, generative, and selective evaluation tasks. You can also design your evaluation scripts inconsistent with the existing scripts.
*  **metric**: This folder contains scripts used to **define the metrics** of the evaluation. Such as precision, bert_score, etc. You devise your own metric or load metric from huggingface.
* **llm**: This folder comprises scripts used to **load LLM** and defines methods for **interacting** with models. You can add your llm here.
* **prompts**: This folder contains some Txt files used as **few-shots prompts** when interacting with llm.
* **core**: This folder includes the core scripts of this project. We design analyst.py for **analyzing experimental data** and save it as CSV files. We also design experiment.py to run experiments with **multiple language models** and evaluators on a given dataset.
# Quick Start

Get started quickly with a 20-line demo program.

* UHGEval requires Python>=3.10.0
* `pip install -r requirements.txt`
* Take `uhgeval/configs/example_config.py` as an example, create `uhgeval/configs/real_config.py` and add configs of models you want to evaluate to this script.
* Run `demo.py`

# Advanced Usage

Utilize `run_uhgeval.py` or `run_uhgeval_future.py` for a comprehensive understanding of this project. The former is currently a **provisional piece** of code slated for removal in the future; whereas the latter is **command-line executable** code intended for future use. Both scripts will perform a complete evaluation process using the GPT model. 
# Customization Guideline
Our project can be customized to your needs. Here are some guidelines to tell you how to carry out this process.
* If you want to add a **new dataset** to this project, you should first add the dataset files like `UHGEval/data/dataset_name/A.json`. Then, add the script used to load this dataset like `UHGEval/uhgeval/dataset/loadyourdataset.py`.
* If you want to add a **new model** to be evaluated, you should first confirm the load method of this model and then add it to the **corresponding location**. For example, if this model is an API-type model, you're supposed to add it like `UHGEval/uhgeval/llm/api.py`.
* If you want to design your **own metric**, you should add it in `UHGEval/uhgeval/metric/common.py`.
* If you want to devise a **new evaluator**, you should first add it like `UHGEval/uhgeval/evaluator/your_evaluator.py`. Please make sure this script is consistent with the `base.py`. Then you need to add the **method** for interacting with models in `UHGEval/uhgeval/llm/base.py`. Don't forget to add prompt files like`UHGEval/uhgeval/prompts/your_prompts.txt`
# Results for Experiment-20231117

<p align="center"><img src="./assets/discri_and_sel.png" alt=""></p>

<p align="center"><img src="./assets/gen.png" alt=""></p>

<p align="center"><img src="./assets/by_type.png" alt="" width="60%"></p>

The original experimental results are in [./archived_experiments/20231117](./archived_experiments/20231117).


# Contributions

Although we have conducted thorough automatic annotation and manual verification, there may still be errors or imperfections in our [XinhuaHallucinations](./data/Xinhua/XinhuaHallucinations.json) dataset with over 5000 data points. We encourage you to raise issues or submit pull requests to assist us in improving the consistency of the dataset. You may also receive corresponding recognition and rewards for your contributions.

## TODOs

<details>
<summary>Click me to show all TODOs</summary>

- [ ] llm, metric: enable loading from HuggingFace
- [ ] config: utilize conifg to realize convenient experiment
- [ ] TruthfulQA: add new dataset and corresponding evaluators
- [ ] another repo: creation pipeline of dataset

</details>

## CITATION

```BibTeX
@article{UHGEval,
    title={UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation},
    author={Xun Liang and Shichao Song and Simin Niu and Zhiyu Li and Feiyu Xiong and Bo Tang and Zhaohui Wy and Dawei He and Peng Cheng and Zhonghao Wang and Haiying Deng},
    journal={arXiv preprint arXiv:2311.15296},
    year={2023},
}
```
