import argparse
import json

from .benchs import load_evaluator
from .llms import HuggingFace, OpenAIAPI
from .logging import logger


# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(description="CLI for Eval Suite.")

    # Common arguments
    parser.add_argument("--model_type", type=str, required=True, choices=["openai_api", "huggingface"])
    parser.add_argument("--generation_configs", type=json.loads, default={}, help="Model generation configs, which can be overridden by evaluators.")
    parser.add_argument("--evaluators", type=str, nargs="+", required=True, help="Shortened dotted paths to the evaluators to run.")

    # OpenAI API model arguments
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str)

    # Hugging Face model arguments
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--apply_chat_template", action="store_true", help="Whether to apply chat template.")

    args = parser.parse_args()

    # Validation
    if args.model_type == "openai_api" and (not args.api_key or not args.base_url):
        parser.error("--model_name and --api_key are required for 'openai_api' model.")
    if args.model_type == "huggingface" and not args.model_name_or_path:
        parser.error("--model_name_or_path is required for 'huggingface' model.")

    return args
# fmt: on


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Start the CLI with args: {args}")

    if args.model_type == "openai_api":
        model = OpenAIAPI(
            model_name=args.model_name,
            generation_configs=args.generation_configs,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    elif args.model_type == "huggingface":
        model = HuggingFace(
            model_name_or_path=args.model_name_or_path,
            generation_configs=args.generation_configs,
            apply_chat_template=args.apply_chat_template,
        )

    evaluator_classes = [load_evaluator(name) for name in args.evaluators]
    for evaluator_class in evaluator_classes:
        evaluator_object = evaluator_class(model)
        evaluator_object.evaluate()
