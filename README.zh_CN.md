[English](./README.md) | [中文简体](./README.zh_CN.md)

<a href="https://opensource.org/license/apache-2-0/">
    <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-yellow.svg">
</a>
<a href="https://arxiv.org/abs/2311.15296">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
</a>

# 目录
- [介绍](#介绍)
- [项目结构](#项目结构)
  * [uhgeval](#uhgeval)
- [快速开始](#快速开始)
- [进阶使用](#进阶使用)
- [定制化指导](#定制化指导)
- [实验-20231117的结果](#实验-20231117的结果)
- [贡献](#贡献)
- [引用](#引用)



# 🍄 介绍

通过无约束生成对中文大型语言模型的幻觉产生进行基准测试

* 安全：确保实验数据的安全性至关重要。
* 灵活：易于扩展，所有模块均可替换。

<p align="center"><img src="./assets/eval_framework.png" alt="" width="80%"></p>
# 项目结构
这个项目包含多个文件夹和多个Python脚本文件。这里给出它们对应的介绍。

* **data**: 该文件夹包括了用于评估的数据集文件。目前该文件夹中只包含了XinhuaHallucinations数据集文件。你可以在这里添加自己的数据集文件。
* **logs**: 该文件夹用于存储日志，警告，报错，信息等，比如token花费数，网络连接问题，没有正确响应等。
* **output**: 该文件夹用于存储评估产生的结果文件。
* **statistics**: 该文件夹用于存储多次实验产生的总结果文件。
* **tests**: 该文件夹包括了用于进行单元测试的脚本文件。
* **uhgeval**: 该文件夹包括了该项目的核心组件。我们会在下方给出一个更详细的介绍。

## uhgeval
该文件夹中所包含的文件确保了项目的正确运行。在这里我们会分别介绍他们的作用。
* **configs**: 该文件夹包含了用于初始化模型加载参数的脚本。在脚本中存储了许多字符串。字符串可以是一个API，一个URL或者本地路径，取决于模型的加载方式。
* **dataset**: 该文件夹包含了用于加载数据集的脚本。目前文件夹中只包含了用于加载XinhuaHallucinations数据集的脚本。你可以在这里添加自己的数据集加载脚本。
* **evaluator**: 该文件夹包含了用于执行评估任务的脚本。这里我们提供了执行判别式评估，选择式评估和生成式评估的脚本。你可以设计自己的评估脚本，注意务必与现有评估脚本保持一致。
*  **metric**: 该文件夹包括了用于定义评估中所使用的指标的脚本，比如正确率，bert_socre等。你可以设计自己的评估指标或者从huggingface上导入。
* **llm**: 该文件夹包括了用于加载大模型和定义模型交互方法的脚本。你可以在这里添加自己的大模型。
* **prompts**: 该文件夹包含了一些TXT文件，在与模型交互它们会被用作执行few-shots的prompt。
* **core**: 该文件夹包括了该项目的核心脚本。我们设计了`analyst.py`用于分析实验数据并将其存储成CSV文件，同时也设计了`experiment.py`来在指定数据集上使用多个语言模型和多种评估方式进行实验。


# 快速开始

通过一个20行的演示程序快速入门。

* UHGEval 需要 Python>=3.10.0
* `pip install -r requirements.txt`
* 以 `uhgeval/configs/example_config.py` 为例，创建 `uhgeval/configs/real_config.py` 在其中添加你要评估的模型的配置信息。
* 运行 `demo.py`

# 进阶使用

使用 `run_uhgeval.py` 或 `run_uhgeval_future.py` 来深入了解该项目。前者是暂时的一个代码，未来会删除掉；后者是通过命令行进行运行的代码，未来会使用该代码。两个脚本都会使用GPT系列模型执行一次完整的评估过程。

# 定制化指南
我们的项目可以根据你的需求进行定制。这里给出一些关于如何进行定制化的指南。
* 如果你想要添加一个新的数据集，你首先需要以这种形式 `UHGEval/data/dataset_name/A.json`添加数据集文件。然后， 以这种形式 `UHGEval/uhgeval/dataset/loadyourdataset.py`添加数据集加载脚本。
* 如果你想要添加一个新的语言模型，你需要首先确定它的加载方式，然后将其添加到对应的脚本文件中。比如，如果某个模型是API类型，你需要在`UHGEval/uhgeval/llm/api.py`中添加你的模型。
* 如果你想要设计自己的评估指标，你需要将其添加到`UHGEval/uhgeval/metric/common.py`中。
* 如果你想设计新的评估方式，你需要首先以这种形式`UHGEval/uhgeval/evaluator/your_evaluator.py`添加评估脚本。请务必确保添加的脚本在结构上与  `base.py`保持一致。然后，你需要在`UHGEval/uhgeval/llm/base.py`中添加对应的交互方法。不要忘记以这种形式`UHGEval/uhgeval/prompts/your_prompts.txt`添加对应的prompts文件。

# 实验-20231117的结果

<p align="center"><img src="./assets/discri_and_sel.png" alt=""></p>

<p align="center"><img src="./assets/gen.png" alt=""></p>

<p align="center"><img src="./assets/by_type.png" alt="" width="60%"></p>

原始实验结果在 [./archived_experiments/20231117](./archived_experiments/20231117)。

# 贡献

虽然我们已经进行了充分的自动标注和人工复检，但有 5000 多个数据项的 [XinhuaHallucinations](./data/Xinhua/XinhuaHallucinations.json) 数据集仍然可能存在错误或不完善之处。我们期待您提出 Issue 或提交 Pull Request，以帮助我们改进数据集的一致性。您还有可能获得相应的认可及奖励。

# 引用

```BibTeX
@article{UHGEval,
    title={UHGEval: Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation},
    author={Xun Liang and Shichao Song and Simin Niu and Zhiyu Li and Feiyu Xiong and Bo Tang and Zhaohui Wy and Dawei He and Peng Cheng and Zhonghao Wang and Haiying Deng},
    journal={arXiv preprint arXiv:2311.15296},
    year={2023},
}
```
