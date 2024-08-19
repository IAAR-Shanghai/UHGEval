import argparse
import json
from pprint import pprint

from .benchs import __all__ as all_evaluators
from .benchs import load_evaluator
from .llms import HuggingFace, OpenAIAPI
from .logging import logger
from .utils import save_stats


# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(description="This CLI for Eval Suite allows you to evaluate LLMs using Hugging Face or OpenAI, save statistics, and list available evaluators.")
    sub_parsers = parser.add_subparsers(help="Choose the operation to run.", dest="operation_name")

    # ─── 1. Conduct Evaluations ───────────────────────────────────────────

    eval_parser = sub_parsers.add_parser("eval", help="Conduct evaluations")
    sub_sub_parsers = eval_parser.add_subparsers(help="Different model loading type", dest="model_type")

    # ─── 1.1 Load Hugging Face Model ──────────────────────────────────────

    hf_paser = sub_sub_parsers.add_parser("huggingface", help="Load Hugging Face model")
    hf_paser.add_argument("--model_name_or_path", type=str)
    hf_paser.add_argument("--generation_configs", type=json.loads, default={}, help="Model generation configs, which can be overridden by evaluators.")
    hf_paser.add_argument("--apply_chat_template", action="store_true", help="Whether to apply chat template.")
    hf_paser.add_argument("--evaluators", type=str, nargs="+", required=True, help="Evaluator names.")

    # ─── 1.2 Load OpenAI Model ────────────────────────────────────────────

    openai_parser = sub_sub_parsers.add_parser("openai", help="Load OpenAI model")
    openai_parser.add_argument("--model_name", type=str)
    openai_parser.add_argument("--generation_configs", type=json.loads, default={}, help="Model generation configs, which can be overridden by evaluators.")
    openai_parser.add_argument("--api_key", type=str)
    openai_parser.add_argument("--base_url", type=str)
    openai_parser.add_argument("--evaluators", type=str, nargs="+", required=True, help="Evaluator names.")

    # ─── 2. Save Statistics On Evaluation Results ─────────────────────────

    stat_parser = sub_parsers.add_parser("stat", help="Save statistics on evaluation results")
    stat_parser.add_argument("--output-dir", type=str, default="./output/", help="The directory containing the output files.")
    stat_parser.add_argument("--stats-path", type=str, default="./stats.csv", help="The path to save the stats.")

    # ─── 3. List All Evaluators ───────────────────────────────────────────

    list_parser = sub_parsers.add_parser("list", help="List all evaluators")

    args = parser.parse_args()
    return args
# fmt: on


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Start the CLI with args: {args}")

    if args.operation_name == "eval":
        evaluator_classes = [load_evaluator(name) for name in args.evaluators]
        if args.model_type == "huggingface":
            model = HuggingFace(
                model_name_or_path=args.model_name_or_path,
                generation_configs=args.generation_configs,
                apply_chat_template=args.apply_chat_template,
            )
        elif args.model_type == "openai":
            model = OpenAIAPI(
                model_name=args.model_name,
                generation_configs=args.generation_configs,
                api_key=args.api_key,
                base_url=args.base_url,
            )
        for evaluator_class in evaluator_classes:
            evaluator_object = evaluator_class(model)
            evaluator_object.evaluate()

    elif args.operation_name == "stat":
        save_stats(output_dir=args.output_dir, stats_path=args.stats_path)

    elif args.operation_name == "list":
        print("All evaluators:")
        pprint(all_evaluators)
