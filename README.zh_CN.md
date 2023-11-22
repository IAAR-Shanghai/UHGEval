[English](./README.md) | [中文简体](./README.zh_CN.md)

# 💡 UHGEval

通过无约束生成对中文大型语言模型的幻觉产生进行基准测试

* 安全：确保实验数据的安全性至关重要。
* 灵活：易于扩展，所有模块均可替换。

![UHG Framework](./assets/eval_framework.png)

## 快速开始

通过一个20行的演示程序快速入门。

* UHGEval 需要 Python>=3.10.0
* `pip install -r requirements.txt`
* 以 `uhgeval/configs/example_config.py` 为例，创建 `uhgeval/configs/real_config.py` 以配置 OpenAI GPT 的密钥。
* 运行 `demo.py`

## 进阶使用

使用 `run_uhgeval.py` 或 `run_uhgeval_future.py` 来深入了解该项目。前者是暂时的一个代码，未来会删除掉；后者是通过命令行进行运行的代码，未来会使用该代码。

## 引用

```BibTeX
@article{UHGEval,
  author = {Xun Liang and Shichao Song and Simin Niu and Zhiyu Li and Feiyu Xiong and Bo Tang and Zhaohui Wy and Dawei He and Peng Cheng and Zhonghao Wang and Haiying Deng},
  title = {UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation},
  year = {2023}
}
```
