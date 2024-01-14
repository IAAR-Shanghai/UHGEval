import importlib
import sys

from loguru import logger
from uhgeval.core.analyst import save_overalls, save_overalls_by_type
from uhgeval.core.experiment import experiment_in_blocks
from uhgeval.core.load_conf import load_yaml_conf

if __name__ == '__main__':
    seed = 22
    enable_log_saving = True
    logger.remove()  # Remove all logger handlers including the stderr logger handler
    logger.add(sys.stderr, level=40)  # Update stderr logger
    logger.add('logs/uhgeval_{time}.log',
               level=0) if enable_log_saving else ...
    # TODO: Currently, loguru does not support log settings above when using
    # the 'spawn' method in multiprocessing.
    experiment_configs_path = 'config.yaml'
    experiment_conf = load_yaml_conf(experiment_configs_path)
    datasets_configs = experiment_conf.get('dataset')
    evaluated_dataset_list = []
    evaluator_list = []
    # Load dataset configs
    for dataset_dict in datasets_configs:
        dataset_name = list(dataset_dict.keys())[0]
        parameters = list(dataset_dict.values())[0]
        temp_name = dataset_name.split('.')
        dataset_module_name = f"uhgeval.dataset.{temp_name[0]}"
        dataset_module = importlib.import_module(dataset_module_name)
        dataset_class = getattr(dataset_module, temp_name[1])
        # Load evaluator configs
        for evaluator_config in parameters['evaluator']:
            temp_list = evaluator_config.split('.')
            evaluator_module_name = f"uhgeval.evaluator.{temp_list[0]}"
            evaluator_module = importlib.import_module(evaluator_module_name)
            evaluator_class = getattr(evaluator_module, temp_list[1])
            # Instantiate the corresponding class and add it to the evaluator list.
            evaluator_list.append(evaluator_class)
        parameters.pop('evaluator')
        # Instantiate the corresponding class and add it to the dataset list.
        if parameters is not None:
            evaluated_dataset_list.append(dataset_class(**parameters))
        else:
            evaluated_dataset_list.append(dataset_class())

    # Load llm configs
    llm_configs = experiment_conf.get('llm')
    evaluated_llm_list = []
    for llm_type, llm_list in llm_configs.items():
        if llm_list is None:
            continue
        for llm_dict in llm_list:
            if llm_dict is None:
                continue
            llm_name = list(llm_dict.keys())[0]
            parameters = list(llm_dict.values())[0]
            llm_module_name = f"uhgeval.llm.{llm_type}"
            llm_module = importlib.import_module(llm_module_name)
            llm_class = getattr(llm_module, llm_name)
            # Instantiate the corresponding class and add it to the llm list.
            if parameters is not None:
                evaluated_llm_list.append(llm_class(**parameters))
            else:
                evaluated_llm_list.append(llm_class())

    for dataset in evaluated_dataset_list:
        experiment_in_blocks(
            dataset,
            evaluated_llm_list,
            evaluator_list,
            3,
            170,
            0,
            seed)
        save_overalls()
        # save_overalls_by_type()
