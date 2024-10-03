import json
import os
from collections import defaultdict

import pandas as pd

from .logging import logger


def save_stats(output_dir: str = "./output/", stats_path: str = "./stats.csv"):
    """Save the overall stats of the outputs to a csv file.

    Args:
        output_dir (str, optional): The directory containing the output files.
        stats_path (str, optional): The path to save the stats.

    Note:
        `overall_dicts` is a dictionary of dictionaries, where the keys are model names
        and the values are dictionaries of overall stats. For example:
        ```
        overall_dicts = {
            "Model1": {"GenerativeEvaluator_bleu": 0.1, "GenerativeEvaluator_rouge": 0.2},
            "Model2": {"GenerativeEvaluator_bleu": 0.2, "GenerativeEvaluator_rouge": 0.3},
        }
        ```
    """

    # ─── 1. Get Output File Paths ─────────────────────────────────────────

    filenames = os.listdir(output_dir)
    paths = [os.path.join(output_dir, name) for name in filenames]
    paths = sorted(paths)

    # ─── 2. Get Overall Dicts ─────────────────────────────────────────────

    overall_dicts = defaultdict(dict)
    for path in paths:

        # ─── 2.1 Load Json Output ─────────────────────────────────────

        if not os.path.exists(path):
            logger.error(f"Path {path} does not exist")
            continue
        if not path.endswith(".json"):
            logger.error(f"Path {path} is not a json file")
            continue
        with open(path, encoding="utf-8") as f:
            output = json.load(f)

        # ─── 2.2 Get Model Name And Overall Dict ──────────────────────

        meta = output.get("meta", {})
        evaluator_name = meta.get("evaluator", {}).get("name", "Evaluator")
        model_name = meta.get("model", {}).get("name", "LLM")
        overall = output.get("overall", {})

        overall = {f"{evaluator_name}_{k}": v for k, v in overall.items()}
        overall_dicts[model_name].update(overall)

    # ─── 3. Save Stats To CSV ─────────────────────────────────────────────

    df = pd.DataFrame.from_dict(overall_dicts, orient="index")
    df.to_csv(stats_path)
    logger.info(f"Stats saved to {stats_path}")
