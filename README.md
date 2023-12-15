[English](./README.md) | [中文简体](./README.zh_CN.md)

<a href="https://opensource.org/license/apache-2-0/">
    <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-yellow.svg">
</a>
<a href="https://arxiv.org/abs/2311.15296">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
</a>

# 🍄 UHGEval

Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation

* Safety: Ensuring the security of experimental data is of utmost importance.
* Flexibility: Easily expandable, with all modules replaceable.

<p align="center"><img src="./assets/eval_framework.png" alt="" width="80%"></p>

## Quick Start

Get started quickly with a 20-line demo program.

* UHGEval requires Python>=3.10.0
* `pip install -r requirements.txt`
* Take `uhgeval/configs/example_config.py` as an example, create `uhgeval/configs/real_config.py` to configure the OpenAI GPT section.
* Run `demo.py`

## Advanced Usage

Utilize `run_uhgeval.py` or `run_uhgeval_future.py` for a comprehensive understanding of this project. The former is currently a provisional piece of code slated for removal in the future; whereas the latter is command-line executable code intended for future use.

## Results for Experiment-20231117

<p align="center"><img src="./assets/discri_and_sel.png" alt=""></p>

<p align="center"><img src="./assets/gen.png" alt=""></p>

<p align="center"><img src="./assets/by_type.png" alt="" width="60%"></p>

The original experimental results are in [./archived_experiments/20231117](./archived_experiments/20231117).

## Contributions

Although we have conducted thorough automatic annotation and manual verification, there may still be errors or imperfections in our [XinhuaHallucinations](./data/Xinhua/XinhuaHallucinations.json) dataset with over 5000 data points. We encourage you to raise issues or submit pull requests to assist us in improving the consistency of the dataset. You may also receive corresponding recognition and rewards for your contributions.

## TODOs

<details>
<summary>Click me to show all TODOs</summary>

- [ ] llm, metric: enable loading from HuggingFace
- [ ] config: utilize conifg to realize convenient experiment
- [ ] TruthfulQA: add new dataset and corresponding evaluators
- [ ] another repo: creation pipeline of dataset
- [ ] contribution: OpenCompass

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
