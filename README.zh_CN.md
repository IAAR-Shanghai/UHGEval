[English](./README.md) | [中文简体](./README.zh_CN.md)

<a href="https://opensource.org/license/apache-2-0/">
    <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-yellow.svg">
</a>
<a href="https://arxiv.org/abs/2311.15296">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
</a>

# 🍄 UHGEval

通过无约束生成对中文大型语言模型的幻觉产生进行基准测试

* 安全：确保实验数据的安全性至关重要。
* 灵活：易于扩展，所有模块均可替换。

<p align="center"><img src="./assets/eval_framework.png" alt="" width="80%"></p>

## 快速开始

通过一个20行的演示程序快速入门。

* UHGEval 需要 Python>=3.10.0
* `pip install -r requirements.txt`
* 以 `uhgeval/configs/example_config.py` 为例，创建 `uhgeval/configs/real_config.py` 以配置 OpenAI GPT 的密钥。
* 运行 `demo.py`

## 进阶使用

使用 `run_uhgeval.py` 或 `run_uhgeval_future.py` 来深入了解该项目。前者是暂时的一个代码，未来会删除掉；后者是通过命令行进行运行的代码，未来会使用该代码。

## 实验-20231117的结果

<p align="center"><img src="./assets/discri_and_sel.png" alt=""></p>

<p align="center"><img src="./assets/gen.png" alt=""></p>

<p align="center"><img src="./assets/by_type.png" alt="" width="60%"></p>

原始实验结果在 [./archived_experiments/20231117](./archived_experiments/20231117)。

## 贡献

虽然我们已经进行了充分的自动标注和人工复检，但有 5000 多个数据项的 [XinhuaHallucinations](./data/Xinhua/XinhuaHallucinations.json) 数据集仍然可能存在错误或不完善之处。我们期待您提出 Issue 或提交 Pull Request，以帮助我们改进数据集的一致性。您还有可能获得相应的认可及奖励。

## 引用

```BibTeX
@article{UHGEval,
    title={UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation},
    author={Xun Liang and Shichao Song and Simin Niu and Zhiyu Li and Feiyu Xiong and Bo Tang and Zhaohui Wy and Dawei He and Peng Cheng and Zhonghao Wang and Haiying Deng},
    journal={arXiv preprint arXiv:2311.15296},
    year={2023},
}
```
