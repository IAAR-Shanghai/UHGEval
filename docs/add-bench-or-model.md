# Customization Guidelines

## Adding a New Benchmark

You can refer to the structure of the `eval/benchs/exampleqa` folder, which serves as a minimal benchmark example. Additionally, you might want to check the `eval/benchs/base_dataset.py` and `eval/benchs/base_evaluator.py` files, as they provide the base classes for benchmarks.

1. **Creating a Benchmark Folder**
   - Create a new folder under the `benchs` directory.
   - The folder should contain the dataset, evaluator, and any other necessary files.
   - The folder should include a `README.md` file to provide an overview of the benchmark and any specific instructions for running it.

2. **Dataset**
   - Ensure the benchmark folder includes a `dataset.py` file, which contains the dataset loading logic.
   - Implement a subclass that inherits from `BaseDataset`.
   - The subclass must implement the `load` method, which returns a `list[dict]`, where each element is an evaluation data sample and must contain a unique `id` field.

3. **Evaluator**
   - The folder must include an evaluator script, typically named `eval_{benchmark_name}.py`, which implements the evaluation logic.
   - This script should inherit from `BaseEvaluator` and contain the benchmark's evaluation logic.
   - The subclass should implement `set_generation_configs` to determine the default token generation settings for the LLM during evaluation.
   - Implement `load_batched_dataset` to load the batched dataset from `dataset.py`.
   - Implement `scoring` to evaluate one data item from the datasets, returning the evaluation result in a dictionary format.
   - Implement `compute_overall` to aggregate the evaluation results into an overall assessment, returning a dictionary with the final evaluation results.

4. **Registering the Benchmark**
   - Add the benchmark to the `__init__.py` file under the `benchs` directory to ensure it is discoverable by the framework.
   - Import the benchmark class in the `__init__.py` file and add it to the `__all__` list.

5. **Documentation**
   - Create a `README.md` file in the benchmark folder.
   - Include any specific instructions or requirements for running the benchmark.
   - Add one line to the `README.md` file in the root directory under the `## Eval Suite` section to introduce the new benchmark.

## Adding a New Model Loader

You can refer to the `eval/llms/huggingface.py` and `eval/llms/openai_api.py` files as examples for loading LLMs.

1. **Language Model Loader**
   - Create a new file under the `llms` directory. 
   - The file should contain the logic for loading the LLM from a specific source (e.g., Hugging Face, OpenAI API).

2. **Implementation Steps**
   - Implement a subclass that inherits from `BaseLLM`.
   - The subclass must implement `update_generation_configs` to handle parameter conversions, as different LLM loaders may have varying parameter names (e.g., `max_tokens` vs. `max_new_tokens`).
   - The subclass must implement `_request`, which accepts a `str` as input and returns a `str` as the generated output.

3. **Registering the LLM Loader**
   - Register the new LLM loader in the `__init__.py` file under the `llms` directory.
   - Import the new LLM loader in the `__init__.py` file and add it to the `__all__` list.
