#!/usr/bin/env python3
"""
Metrics computed from the logs

- Forgetting
- CumulativeForgetting
- WC_Acc
- Min_Acc

"""

import numpy as np
import pandas as pd

from src.toolkit.process_results import extract_results


def compute_forgetting(dataframe, metric_name, prefix="Forgetting_"):
    """
    Compute online forgetting for one task-specific metric.

    For each seed, forgetting at a measurement point is the maximum accuracy
    observed so far for that task minus the current accuracy.  Future
    observations are never used, so the result is causal and cannot leak
    information from later training stages.
    """

    df = dataframe.sort_values("mb_index").copy()

    if "valid_stream" in metric_name:
        raise NotImplementedError(
            "The computation of forgetting on continual metric streams is not supported"
        )
        df = decorate_with_training_task(
            df, base_name="Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp"
        )
    else:
        df = decorate_with_training_task(
            df, base_name="Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp"
        )

    df = df.dropna(subset=["training_exp"])
    df = df.sort_values(by=["seed", "mb_index"])

    # A task metric is NaN until that task has been evaluated.  Forward-fill
    # those gaps, then compare the current value with the best value observed
    # up to this point.  This is the standard max-past-accuracy definition.
    def _forgetting(series):
        observed = series.ffill()
        best_so_far = observed.cummax()
        return best_so_far - observed

    df[prefix + metric_name] = df.groupby("seed", group_keys=False)[metric_name].transform(
        _forgetting
    )

    # The first measurement cannot represent forgetting because no prior
    # post-learning accuracy exists yet.
    first_index = df.groupby("seed").head(1).index
    df.loc[first_index, prefix + metric_name] = np.nan

    return df


def compute_average(dataframe, metric_names, avg_metric_name):
    """
    Computes average of several metric names given

    Adds it as a column
    """

    df = dataframe.sort_values("mb_index")
    df[avg_metric_name] = df[metric_names].mean(axis=1)

    return df


def compute_average_forgetting(
    dataframe,
    num_exp,
    base_name="Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp",
    name="Average_Forgetting",
):
    """
    Compute average forgetting across all task-specific test accuracies.

    At each measurement point, only tasks that have already produced an
    evaluation are included in the mean.  The final value therefore averages
    the forgetting of all learned tasks.
    """

    prefix = "Forgetting_"
    metric_names = []
    for i in range(num_exp):
        mname = base_name + f"{i:03d}"
        dataframe = compute_forgetting(dataframe, mname)
        metric_names.append(prefix + mname)

    df = compute_average(dataframe, metric_names, name)
    return df


def annotate_splits(dataframe, splits):
    for task, (sp1, sp2) in splits.items():
        df.loc[(df.mb_index > sp1) & (df.mb_index <= sp2), "training_exp"] = task
    return df


def compute_min_acc(dataframe, num_tasks=20):
    df = decorate_with_training_task(dataframe, num_tasks=20)
    for task in range(num_tasks):
        metric_name = f"Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task:03d}"
        mask = df["training_exp"] - 1 == task
        df["Min_" + metric_name] = df.groupby("seed", group_keys=False)[
            metric_name
        ].apply(lambda x: x.mask(mask).cummin())
    return df


def compute_AAA(
    dataframe,
    base_name="Top1_Acc_Stream/eval_phase/valid_stream/Task000",
):
    """
    Computes average anytime accuracy
    """
    df = dataframe.sort_values(["seed", "mb_index"])
    df["cumulative_sum"] = df.groupby("seed")[base_name].cumsum()
    df["count"] = df.groupby("seed").cumcount() + 1
    df["AAA"] = df["cumulative_sum"] / df["count"]
    return df


def decorate_with_training_task(
    dataframe,
    num_tasks=20,
    base_name="Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp",
):
    metric_list = [f"{base_name}{i:03d}" for i in range(num_tasks)]
    dataframe["training_exp"] = dataframe[metric_list].count(axis=1)
    return dataframe


def compute_calibrated_accuracy(dataframe):
    """
    Computes average anytime accuracy
    """
    df = decorate_with_training_task(dataframe)
    df["calibrated_accuracy"] = (
        df["Top1_Acc_Stream/eval_phase/valid_stream/Task000"] * df["training_exp"] / 20
    )
    return df


def compute_wcacc(dataframe, num_tasks=20):
    """
    Computes WC-Acc
    """
    df = compute_min_acc(dataframe, num_tasks)
    df = df[~(df.training_exp == 0)]
    df = df.reset_index()

    average_cols = [
        f"Min_Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task:03d}"
        for task in range(num_tasks)
    ]

    exclude_cols = [
        f"Min_Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task-1:03d}"
        for task in df["training_exp"]
    ]

    for i, row in df.iterrows():
        df.loc[i, exclude_cols[i]] = np.nan

    filter_cols = [
        f"Top1_Acc_Exp/eval_phase/valid_stream/Task000/Exp{task-1:03d}"
        for task in df["training_exp"]
    ]

    df["average_min_acc"] = df[average_cols].mean(axis=1)

    for i, row in df.iterrows():
        df.loc[i, "current_task_acc"] = df.loc[i, filter_cols[i]]

    df["WCAcc"] = (
        df["current_task_acc"] * (1 / df["training_exp"])
        + (1 - 1 / df["training_exp"]) * df["average_min_acc"]
    )
    return df


def compute_mean_std_metric(dataframe, metric_name, final_step=True):
    df = dataframe
    max_index = df["mb_index"].max()
    mean = df.loc[df["mb_index"] == max_index, metric_name].mean()
    std = df.loc[df["mb_index"] == max_index, metric_name].std()
    return mean, std


if __name__ == "__main__":
    results_dir = "/DATA/ocl_survey/er_split_cifar100_20_2000/"

    frames = extract_results(results_dir)
    df = frames["training"]

    df = compute_forgetting(df, "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000")
