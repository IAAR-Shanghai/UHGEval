# Eval Suite in Detail

## Architecture

TODO

## Project Structure

```bash
eval
├── __init__.py
├── cli.py                              # Command line interface
├── logging.py                          # Global logging configuration
├── metrics.py                          # Common metrics
├── benchs                              # Benchmarks
│   ├── __init__.py
│   ├── base_dataset.py                 # Base class for dataset
│   ├── base_evaluator.py               # Base class for evaluator
│   ├── exampleqa                       # ExampleQA benchmark (a minimal example for reference)
│   │   ├── README.md                   # (Required) README for understanding and using the benchmark
│   │   ├── dataset.py                  # (Required) Loads the dataset for this benchmark
│   │   ├── dataset_exampleqa.jsonl     # (Optional) Source dataset file
│   │   └── eval_exampleqa.py           # (Required) Evaluator to perform the evaluation task
│   ├── halluqa                         # HalluQA Benchmark
│   │   └── ......
│   ├── halueval                        # HaluEval Benchmark
│   │   └── ......
│   └── uhgeval                         # UHGEval Benchmark
│   │   └── ......
│   └── ......
└── llms                                # Language models
    ├── __init__.py
    ├── base_llm.py                     # Base class for language model
    ├── huggingface.py                  # Load LLM from Hugging Face
    └── openai_api.py                   # Load LLM with OpenAI-Compatible API
```
