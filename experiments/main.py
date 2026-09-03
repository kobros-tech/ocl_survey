#!/usr/bin/env python3
import json
import os

import hydra
import omegaconf
import torch

import src.factories.benchmark_factory as benchmark_factory
import src.factories.method_factory as method_factory
import src.factories.model_factory as model_factory
import src.toolkit.utils as utils
from avalanche.benchmarks import with_classes_timeline
from avalanche.benchmarks.scenarios.online import split_online_stream
from src.factories.benchmark_factory import DS_SIZES
from src.strategies.skill_memory import SkillMemoryPlugin


@hydra.main(config_path="../config", config_name="config.yaml")
def main(config):
    utils.set_seed(config.experiment.seed)

    # Keep the experiment portable across local CPU runs and GPU-backed
    # environments such as Google Colab. Explicit device values are preserved;
    # only the "auto" setting is resolved here at runtime.
    configured_device = str(config.strategy.device).lower()
    if configured_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config.strategy.device = device
    else:
        device = str(config.strategy.device)

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    plugins = []

    scenario = benchmark_factory.create_benchmark(
        **config["benchmark"].factory_args,
        dataset_root=config.benchmark.dataset_root,
    )

    model = model_factory.create_model(
        **config["model"],
        input_size=DS_SIZES[config.benchmark.factory_args.benchmark_name],
    )

    optimizer, scheduler_plugin = model_factory.get_optimizer(
        model,
        optimizer_type=config.optimizer.type,
        scheduler_type=config.scheduler.type,
        kwargs_optimizer=config["optimizer"],
        kwargs_scheduler=config["scheduler"],
    )
    print(optimizer)

    if scheduler_plugin is not None:
        plugins.append(scheduler_plugin)

    exp_name = (
        config.strategy.name
        + "_"
        + config.benchmark.factory_args.benchmark_name
        + "_"
        + str(config.benchmark.factory_args.n_experiences)
        + "_"
        + str(config.strategy.mem_size)
    )

    if not config.experiment.debug:
        logdir = os.path.join(
            str(config.experiment.results_root),
            exp_name,
            str(config.experiment.seed),
        )
    else:
        logdir = os.path.join(
            str(config.experiment.results_root),
            "debug",
        )

    if config.experiment.logdir is None:
        os.makedirs(logdir, exist_ok=True)
        utils.clear_tensorboard_files(logdir)

        # Add full results dir to config
        config.experiment.logdir = logdir

        omegaconf.OmegaConf.save(config, os.path.join(logdir, "config.yaml"))
    else:
        logdir = config.experiment.logdir

    strategy = method_factory.create_strategy(
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        logdir=logdir,
        name=config.strategy.name,
        dataset_name=config.benchmark.factory_args.benchmark_name,
        strategy_kwargs=config["strategy"],
        evaluation_kwargs=config["evaluation"],
        experiment_seed=int(config.experiment.seed),
    )

    print("Using strategy: ", strategy.__class__.__name__)
    print("With plugins: ", strategy.plugins)

    skill_memory_plugin = next(
        (plugin for plugin in strategy.plugins if isinstance(plugin, SkillMemoryPlugin)),
        None,
    )

    # Forced REUSE/CLONE/SCRATCH is a mechanism-level ablation, not part of the
    # adaptive benchmark policy. Apply it after factory construction so the
    # existing OCL Survey strategy lifecycle remains authoritative.
    if skill_memory_plugin is not None:
        forced_decision = config.strategy.get("force_decision", None)
        if forced_decision is not None:
            forced_decision = str(forced_decision).lower()
            if forced_decision not in (
                SkillMemoryPlugin.REUSE,
                SkillMemoryPlugin.CLONE,
                SkillMemoryPlugin.SCRATCH,
            ):
                raise ValueError(
                    "strategy.force_decision must be one of: reuse, clone, scratch"
                )
            skill_memory_plugin.force_decision = forced_decision
            print(f"Skill Memory forced decision: {forced_decision}")

    for t, experience in enumerate(scenario.train_stream):
        if config.experiment.train_online:
            # Avalanche 0.6's online splitter returns OnlineCLExperience,
            # which intentionally strips classification decorators. Restore
            # the classes timeline before handing the stream to the strategy,
            # since IncrementalClassifier relies on classes_in_this_experience.
            train_stream = split_online_stream(
                [experience],
                experience_size=config.strategy.train_mb_size,
                shuffle=True,
                drop_last=False,
                access_task_boundaries=config.strategy.use_task_boundaries,
            )
            train_stream = with_classes_timeline(train_stream)
        else:
            train_stream = [experience]

        strategy.train(
            train_stream,
            eval_streams=[scenario.valid_stream[: t + 1]],
            num_workers=0,
            drop_last=True,
            reset_optimizer_state=False,
        )

        if config.experiment.save_models:
            torch.save(
                strategy.model.state_dict(),
                os.path.join(logdir, f"model_{t}.pth"),
            )

        if skill_memory_plugin is not None:
            audit_path = os.path.join(logdir, "skill_memory_audit.json")
            with open(audit_path, "w", encoding="utf-8") as audit_file:
                json.dump(skill_memory_plugin.audit_log, audit_file, indent=2)


if __name__ == "__main__":
    main()
