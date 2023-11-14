[English](./README.md) | [中文简体](./README.zh_CN.md)

# UHGEval

通过无约束生成对中文大型语言模型的幻觉产生进行基准测试

特点：
* 安全性：确保实验数据的安全性至关重要。
* 灵活性：易于扩展，所有模块均可替换。

## 快速开始

通过一个 20 行的演示程序快速入门。

* `pip install -r requirements.txt`
* 以 `uhgeval/configs/example_config.py` 为例，创建 `uhgeval/configs/real_config.py` 来配置 OpenAI GPT 部分。
* 运行 `demo.py`

## 进阶使用

运行 `experiment.py` 以深入了解该项目。在试运行过程中，您可能会遇到各种问题，因为该项目目前只是一个演示。目前，您可以按照解释器的提示使程序正常运行。
