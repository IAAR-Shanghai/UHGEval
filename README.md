# UHGEval

Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation

特点：
* 安全：实验数据不丢失是最重要的事情
* 灵活：易于拓展，全部模块均可替换

## 快速开始

通过一个 10 行的 DEMO 程序来快速开始。

* `pip install -r requirements.txt`
* 以 `configs/example_config.py` 为例，编写 `configs/real_config.py`，完成 OpenAI GPT 部分的配置
* 运行 `demo.py`

## 进阶使用

运行 `experiment.py` 以深入了解该项目。您在尝试运行过程中，可能会遇到诸多问题，这是因为该项目目前只是一个 DEMO，暂时您可以根据解释器的提示使程序正常运行。

## TODOs

feat
- [x] 加上模型参数量
- [x] llm 应该支持更新属性
- [x] 重新自己实现rouge和bleu，或者再次测试缓存的问题
- [ ] 自动load模型，简单的比如7b
- [ ] log 从输出中分离
- [x] experiment改进，传进去的应该是实例化之后的
- [x] 数据集做成字典类型的：不做了，如果是字典类型的，id最好就不在字典的value里了，但是这样的话，数据集做split之后就不好确定其id了，总的来说做成字典类型的弊大于利。

bug
- [x] 重大BUG，数据集中的日期对不上：更改filename为id，并且删除其中的日期

future
- [ ] evaluator 中间加一个 hallucinationevaluator 类
- [ ] 抽象类和子类文件分离
- [ ] 分离出prompt文件
- [x] evaluator尽量能够更加通用，以应对异质的数据集：只要求数据集是json格式，内容是字典的列表，每个字典是一个数据项，必须包含一个id字段即可
