#!/usr/bin/env python3

import os
from typing import List, Optional
from copy import deepcopy

import kornia.augmentation as K
import ray
import torch
import torch.nn as nn

import avalanche.logging as logging
import src.toolkit.utils as utils
from avalanche.evaluation.metrics import (StreamTime, accuracy_metrics,
                                          loss_metrics)
from avalanche.models import SCRModel
from avalanche.training.plugins import (EarlyStoppingPlugin, MIRPlugin,
                                        RARPlugin, ReplayPlugin,
                                        SupervisedPlugin)
from avalanche.training.plugins.evaluation import (EvaluationPlugin,
                                                   default_evaluator)
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.supervised import *
from avalanche.training.supervised.mer import MER
from src.factories.benchmark_factory import DS_CLASSES, DS_SIZES
from src.strategies import (ER_ACE, AGEMPlugin, LwFPlugin, OnlineICaRL,
                            OnlineICaRLLossPlugin, SkillMemoryPlugin,
                            make_compatibility, SkillMemory)
from src.toolkit.cumulative_accuracies import CumulativeAccuracyPluginMetric
from src.toolkit.json_logger import JSONLogger
from src.toolkit.lambda_scheduler import LambdaScheduler
from src.toolkit.metrics import ClockLoggingPlugin, TimeSinceStart
from src.toolkit.parallel_eval import ParallelEvaluationPlugin
from src.toolkit.probing import ProbingPlugin
from src.toolkit.review_trick import ReviewTrickPlugin
from src.toolkit.sklearn_probing import SKLearnProbingPlugin


"""Method Factory"""


def create_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    plugins: Optional[List[SupervisedPlugin]] = None,
    logdir: str = None,
    name: str = None,
    dataset_name: str = None,
    evaluation_kwargs=None,
    strategy_kwargs=None,
    experiment_seed: int = 0,
):
    strategy_dict = {
        "model": model,
        "optimizer": optimizer,
        "evaluator": None,
    }
    strategy_args = utils.extract_kwargs(
        ["train_mb_size", "train_epochs", "eval_mb_size", "device"], strategy_kwargs
    )

    if "batch_size_mem" in strategy_kwargs:
        batch_size_mem = strategy_kwargs["batch_size_mem"]
        if batch_size_mem is None:
            strategy_kwargs["batch_size_mem"] = strategy_args["train_mb_size"]

    strategy_dict.update(strategy_args)

    strategy_eval_args = utils.extract_kwargs(
        ["eval_every", "peval_mode"], evaluation_kwargs
    )
    strategy_dict.update(strategy_eval_args)

    if name == "er":
        strategy = "Naive"
        specific_args = utils.extract_kwargs(["mem_size", "batch_size_mem"], strategy_kwargs)
        storage_policy = ClassBalancedBuffer(max_size=specific_args["mem_size"], adaptive_size=True)
        plugins.append(ReplayPlugin(**specific_args, storage_policy=storage_policy))

    elif name == "der":
        strategy = "DER"
        last_layer_name, in_features = utils.get_last_layer_name(model)
        setattr(model, last_layer_name, nn.Linear(in_features, DS_CLASSES[dataset_name]))
        specific_args = utils.extract_kwargs(["alpha", "beta", "mem_size", "batch_size_mem"], strategy_kwargs)
        strategy_dict.update(specific_args)

    elif name == "mir":
        strategy = "Naive"
        specific_args = utils.extract_kwargs(["mem_size", "subsample", "batch_size_mem"], strategy_kwargs)
        plugins.append(MIRPlugin(**specific_args))

    elif name == "er_ace":
        strategy = "ER_ACE"
        specific_args = utils.extract_kwargs(["alpha", "mem_size", "batch_size_mem"], strategy_kwargs)
        strategy_dict.update(specific_args)

    elif name == "er_lwf":
        strategy = "Naive"
        specific_args_replay = utils.extract_kwargs(["mem_size", "batch_size_mem"], strategy_kwargs)
        specific_args_lwf = utils.extract_kwargs(["alpha", "temperature"], strategy_kwargs)
        storage_policy = ClassBalancedBuffer(max_size=specific_args_replay["mem_size"], adaptive_size=True)
        plugins.append(ReplayPlugin(**specific_args_replay, storage_policy=storage_policy))
        plugins.append(LwFPlugin(**specific_args_lwf))

    elif name == "scr":
        strategy = "SCR"
        last_layer_name, in_features = utils.get_last_layer_name(model)
        setattr(model, last_layer_name, torch.nn.Identity())
        projection_network = torch.nn.Sequential(torch.nn.Linear(in_features, in_features), torch.nn.ReLU(inplace=True), torch.nn.Linear(in_features, 128))
        model = SCRModel(feature_extractor=model, projection=projection_network)
        strategy_dict["model"] = model
        specific_args = utils.extract_kwargs(["mem_size", "temperature", "batch_size_mem"], strategy_kwargs)
        if "criterion" in strategy_dict:
            strategy_dict.pop("criterion")
        scr_transforms = torch.nn.Sequential(
            K.RandomResizedCrop(size=(DS_SIZES[dataset_name][0], DS_SIZES[dataset_name][0]), scale=(0.2, 1.0)),
            K.RandomHorizontalFlip(), K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8), K.RandomGrayscale(p=0.2),
        )
        specific_args["augmentations"] = scr_transforms
        strategy_dict.update(specific_args)

    elif name == "icarl":
        strategy = "OnlineICaRL"
        if "criterion" in strategy_dict:
            strategy_dict.pop("criterion")
        strategy_dict.pop("peval_mode")
        model = strategy_dict.pop("model")
        last_layer_name, in_features = utils.get_last_layer_name(model)
        classifier = getattr(model, last_layer_name)
        setattr(model, last_layer_name, nn.Identity())
        strategy_dict["feature_extractor"] = model
        strategy_dict["classifier"] = classifier
        specific_args = utils.extract_kwargs(["mem_size", "lmb"], strategy_kwargs)
        strategy_dict.update(specific_args)
        strategy_dict["criterion"] = OnlineICaRLLossPlugin(strategy_dict.pop("lmb"))

    elif name == "rar":
        strategy = "Naive"
        specific_args = utils.extract_kwargs(["mem_size", "batch_size_mem", "opt_lr", "beta_coef", "decay_factor_fgsm", "epsilon_fgsm", "iter_fgsm"], strategy_kwargs)
        storage_policy = ClassBalancedBuffer(max_size=specific_args["mem_size"], adaptive_size=True)
        last_layer_name, _ = utils.get_last_layer_name(model)
        plugins.append(RARPlugin(**specific_args, name_ext_layer=last_layer_name, storage_policy=storage_policy))

    elif name == "mer":
        strategy = "MER"
        strategy_dict.update(utils.extract_kwargs(["mem_size", "batch_size_mem", "n_inner_steps", "beta", "gamma"], strategy_kwargs))

    elif name == "agem":
        strategy = "Naive"
        plugins.append(AGEMPlugin(**utils.extract_kwargs(["mem_size", "sample_size"], strategy_kwargs)))

    elif name == "gdumb":
        strategy = "GDumb"
        strategy_dict["criterion"] = nn.CrossEntropyLoss()
        strategy_dict.update(utils.extract_kwargs(["mem_size"], strategy_kwargs))

    elif name == "skill_memory":
        strategy = "Naive"
        # Use a frozen copy of the freshly initialized architecture for probe scoring.
        model_template = deepcopy(model)
        skill_memory = SkillMemory(max_skills=int(strategy_kwargs.get("max_skills", 20)))
        compatibility = make_compatibility(
            model_factory=lambda: deepcopy(model_template),
            num_classes=DS_CLASSES[dataset_name],
            probe_samples=int(strategy_kwargs.get("probe_samples", 64)),
            probe_seed=int(experiment_seed),
        )
        skill_plugin = SkillMemoryPlugin(
            memory=skill_memory,
            compatibility=compatibility,
            max_skills=int(strategy_kwargs.get("max_skills", 20)),
            reuse_threshold=float(strategy_kwargs.get("reuse_threshold", 0.90)),
            clone_threshold=float(strategy_kwargs.get("clone_threshold", 0.15)),
            reuse_accuracy_threshold=float(strategy_kwargs.get("reuse_accuracy_threshold", 0.90)),
            clone_accuracy_threshold=float(strategy_kwargs.get("clone_accuracy_threshold", 0.15)),
        )
        plugins.append(skill_plugin)

    elif name == "linear_probing":
        strategy = "Cumulative"
        strategy_dict.pop("peval_mode")
        strategy_dict["criterion"] = nn.CrossEntropyLoss()
        new_logdir = os.path.join(logdir, "linear_probing")
        if not os.path.isdir(new_logdir):
            os.mkdir(new_logdir)
        evaluator, parallel_eval_plugin = create_evaluator(logdir=new_logdir, **evaluation_kwargs)
        strategy_dict.update({"evaluator": evaluator})
        plugins.append(SKLearnProbingPlugin(logdir, prefix="model"))

    if name == "er_with_review":
        strategy = "Naive"
        specific_args = utils.extract_kwargs(["mem_size", "batch_size_mem"], strategy_kwargs)
        storage_policy = ClassBalancedBuffer(max_size=specific_args["mem_size"], adaptive_size=True)
        plugins.append(ReplayPlugin(**specific_args, storage_policy=storage_policy))
        plugins.append(ReviewTrickPlugin(storage_policy=storage_policy, num_epochs=5))

    if strategy_dict["evaluator"] is None:
        evaluator, parallel_eval_plugin = create_evaluator(logdir=logdir, **evaluation_kwargs)
        strategy_dict.update({"evaluator": evaluator})

    if parallel_eval_plugin is not None:
        strategy_dict["eval_every"] = -1
        plugins.append(parallel_eval_plugin)

    return globals()[strategy](**strategy_dict, plugins=plugins)


def get_loggers(loggers_list, logdir, prefix="logs"):
    loggers = []
    if loggers_list is None:
        return loggers
    for logger in loggers_list:
        if logger == "interactive":
            loggers.append(logging.InteractiveLogger())
        if logger == "tensorboard":
            loggers.append(logging.TensorboardLogger(logdir))
        if logger == "text":
            loggers.append(logging.TextLogger(open(os.path.join(logdir, f"{prefix}.txt"), "w")))
        if logger == "json":
            path = os.path.join(logdir, f"{prefix}.json")
            if os.path.isfile(path):
                os.remove(path)
            loggers.append(JSONLogger(path, autoupdate=False))
    return loggers


def get_metrics(metric_names):
    metrics = []
    for m in metric_names:
        if m == "accuracy_metrics":
            metrics.append(accuracy_metrics(stream=True, experience=True))
            metrics.append(accuracy_metrics(stream=True))
        elif m == "loss_metrics":
            metrics.append(loss_metrics(stream=True, experience=True))
            metrics.append(loss_metrics(stream=True))
            metrics.append(loss_metrics(epoch=True))
        elif m == "cumulative_accuracy":
            metrics.append(CumulativeAccuracyPluginMetric())
        elif m == "time":
            metrics.append(TimeSinceStart())
        elif m == "clock":
            metrics.append(ClockLoggingPlugin())
        else:
            metrics.append(globals()[m](stream=True))
    return metrics


def create_evaluator(metrics, logdir, loggers_strategy=None, loggers_parallel=None, parallel_evaluation=False, **parallel_eval_kwargs):
    strategy_metrics = get_metrics(metrics)
    loggers_strategy = get_loggers(loggers_strategy, logdir, prefix="logs")
    evaluator_strategy = EvaluationPlugin(*strategy_metrics, loggers=loggers_strategy)
    parallel_eval_plugin = None
    if parallel_evaluation:
        ray.init(num_gpus=1, num_cpus=12, include_dashboard=True)
        loggers_parallel = get_loggers(loggers_parallel, logdir, prefix="logs_continual")
        parallel_eval_plugin = ParallelEvaluationPlugin(metrics=strategy_metrics, loggers=loggers_parallel, **parallel_eval_kwargs)
    return evaluator_strategy, parallel_eval_plugin
