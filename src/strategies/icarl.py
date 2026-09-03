import copy
from typing import Callable, List, Optional, Union

import torch
from torch.nn import BCELoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.models import NCMClassifier, TrainEvalModel
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.training.templates import SupervisedTemplate


class OnlineICaRLLossPlugin(SupervisedPlugin):
    """iCaRL distillation loss plugin."""

    def __init__(self, lmb: float = 1.0):
        super().__init__()
        self.criterion = BCELoss()
        self.old_classes = set()
        self.new_classes = set()
        self.old_model = None
        self.old_logits = None
        self.lmb = lmb

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with torch.no_grad():
                self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)
        one_hot = torch.zeros(
            targets.shape[0], logits.shape[1], dtype=torch.float, device=logits.device
        )
        one_hot[range(len(targets)), targets.long()] = 1

        if self.old_logits is not None:
            old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, list(self.old_classes)] = old_predictions[:, list(self.old_classes)]
            one_hot_new = one_hot[:, list(self.new_classes)]
            one_hot_old = one_hot[:, list(self.old_classes)]
            predictions_new = predictions[:, list(self.new_classes)]
            predictions_old = predictions[:, list(self.old_classes)]
            self.old_logits = None
            return (
                self.criterion(predictions_new, one_hot_new)
                + self.lmb * self.criterion(predictions_old, one_hot_old)
            ) / 2
        return self.criterion(predictions, one_hot)

    def before_training_exp(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter != 0:
            self.old_classes = self.old_classes.union(self.new_classes)
            self.new_classes = set()
            self.old_model = copy.deepcopy(strategy.model)
            self.old_model.eval()
        self.new_classes = self.new_classes.union(
            strategy.experience.classes_in_this_experience
        )


class OnlineICaRL(SupervisedTemplate):
    """iCaRL strategy without task identities or herding."""

    def __init__(
        self,
        feature_extractor: Module,
        classifier: Module,
        optimizer: Optimizer,
        mem_size: int = 200,
        momentum: float = 0.1,
        batch_size_mem: int = None,
        criterion=OnlineICaRLLossPlugin(),
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[EvaluationPlugin, Callable[[], EvaluationPlugin]] = default_evaluator,
        eval_every=-1,
    ):
        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier,
            eval_classifier=NCMClassifier(normalize=True),
        )
        storage_policy = ClassBalancedBuffer(mem_size, adaptive_size=True)
        replay_plugin = ReplayPlugin(
            mem_size,
            batch_size=train_mb_size,
            batch_size_mem=batch_size_mem,
            storage_policy=storage_policy,
        )
        icarl = _ICaRLPlugin(replay_plugin, momentum)
        if plugins is None:
            plugins = [icarl, replay_plugin]
        else:
            plugins += [icarl, replay_plugin]
        if isinstance(criterion, SupervisedPlugin):
            plugins += [criterion]
        super().__init__(
            model,
            optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )


class _ICaRLPlugin(SupervisedPlugin):
    def __init__(self, replay_plugin, momentum: float = 1.0, num_batch_update=-1):
        super().__init__()
        self.replay_plugin = replay_plugin
        self.momentum = momentum
        self.num_batch_update = num_batch_update

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.eval()
        self.compute_class_means(strategy)
        strategy.model.train()

    @torch.no_grad()
    def compute_class_means(self, strategy):
        class_means = {}
        for dataset in self.replay_plugin.storage_policy.buffer_datasets:
            dl = DataLoader(
                dataset.eval(),
                shuffle=False,
                batch_size=strategy.eval_mb_size,
                drop_last=False,
            )
            num_els = 0
            for x, y, _ in dl:
                num_els += x.size(0)
                label = y[0].item()
                out = strategy.model.feature_extractor(x.to(strategy.device))
                out = torch.nn.functional.normalize(out, p=2, dim=1)
                if label in class_means:
                    class_means[label] += out.sum(0).cpu().detach().clone()
                else:
                    class_means[label] = out.sum(0).cpu().detach().clone()
            if num_els > 0:
                class_means[label] /= float(num_els)
                class_means[label] /= class_means[label].norm()
        strategy.model.eval_classifier.update_class_means_dict(class_means)
