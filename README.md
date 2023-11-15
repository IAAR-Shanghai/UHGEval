[English](./README.md) | [中文简体](./README.zh_CN.md)

# UHGEval

Benchmarking the Hallucination of Chinese Large Language Models via Unconstrained Generation

Features:
* Safety: Ensuring the security of experimental data is of utmost importance.
* Flexibility: Easily expandable, with all modules replaceable.

## Quick Start

Get started quickly with a 20-line demo program.

* UHGEval requires Python>=3.10.0
* `pip install -r requirements.txt`
* Take `uhgeval/configs/example_config.py` as an example, create `uhgeval/configs/real_config.py` to configure the OpenAI GPT section.
* Run `demo.py`

## Advanced Usage

Run `experiment.py` to delve deeper into the project. You may encounter various issues during the trial run because the project is currently just a demo. For now, you can follow the interpreter's prompts to get the program running smoothly.

## TODOs

<details>
<summary>Click me to show all TODOs</summary>

- [x] requirements.txt: add version specifications
- [x] evaluator: add a function, `set_llm()`, to update the llm parameter
- [x] translate: use English throughout
- [ ] docs: update all documentation
- [ ] llm, metric: enable loading from HuggingFace
- [x] running.log: enable log saving
- [ ] evaluator: add `XinhuaHallucinationsEvaluator` class between the abstract class and concrete classes
- [ ] evaluator: optimize the design of `BaseEvaluator`
- [ ] remote.py, gpt.py: Baichuan2_53B_Chat + GPT -> api.py
- [ ] config: utilize conifg to realize convenient experiment

</details>
