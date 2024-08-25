# Experiments

## Experiment-20231117

The original experimental results can be found in [./experiments/20231117](./experiments/20231117).

These are the experimental results corresponding to the [ACL 2024 paper](https://aclanthology.org/2024.acl-long.288/). All evaluations were conducted on the full version of the UHGEvalDataset.

<p align="center"><img src="./experiments/20231117/images/discri_and_sel.png" alt=""></p>

<p align="center"><img src="./experiments/20231117/images/gen.png" alt=""></p>

<p align="center"><img src="./experiments/20231117/images/by_type.png" alt="" width="60%"></p>

> [!Caution]
> The Eval Suite used at that time was an older version. Running the same experiments with the current version might produce slightly different results.

## Experiment-20240822

The original experimental results can be found in [./experiments/20240822](./experiments/20240822).

Tested whether there would be significant differences in the evaluation results produced using the full dataset versus the concise dataset.

The experimental results show that the differences between the full and concise datasets are minimal, so the concise dataset can be used instead of the full dataset to improve evaluation speed.
