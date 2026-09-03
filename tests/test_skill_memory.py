import torch
import pytest
from torch import nn

from src.strategies.skill_memory import SkillMemory, SkillMemoryPlugin, make_probe


class DummyStrategy:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        self.train_epochs = 2


def test_memory_stores_independent_cpu_copy():
    memory = SkillMemory(max_skills=2)
    source = {"weight": torch.tensor([[1.0, 2.0]])}
    memory.register("a", source)

    source["weight"].fill_(9.0)
    assert torch.equal(memory._records["a"].state_dict["weight"], torch.tensor([[1.0, 2.0]]))
    assert memory._records["a"].state_dict["weight"].device.type == "cpu"


def test_memory_capacity_is_bounded():
    memory = SkillMemory(max_skills=1)
    memory.register("a", {"weight": torch.ones(1)})
    with pytest.raises(RuntimeError, match="at capacity"):
        memory.register("b", {"weight": torch.zeros(1)})


def test_best_match_selects_highest_score():
    memory = SkillMemory(max_skills=3)
    memory.register("a", {"weight": torch.ones(1)})
    memory.register("b", {"weight": torch.zeros(1)})

    def scorer(record, _query):
        return 0.2 if record.name == "a" else 0.8

    record, score = memory.best_match(None, scorer)
    assert record.name == "b"
    assert score == pytest.approx(0.8)


def test_optimizer_is_rebound_to_current_model_parameters():
    model = nn.Sequential(nn.Linear(2, 2))
    strategy = DummyStrategy(model)
    plugin = SkillMemoryPlugin()

    old_parameters = list(strategy.optimizer.param_groups[0]["params"])
    model[0] = nn.Linear(2, 3)
    plugin._reset_optimizer(strategy)

    new_parameters = list(model.parameters())
    assert list(strategy.optimizer.param_groups[0]["params"]) == new_parameters
    assert all(p not in new_parameters for p in old_parameters)


def test_probe_is_deterministic_for_same_seed():
    class Experience:
        pass

    experience = Experience()
    experience.dataset = [(torch.tensor([float(i)]), i) for i in range(10)]

    x1, y1 = make_probe(experience, samples=5, seed=123)
    x2, y2 = make_probe(experience, samples=5, seed=123)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_probe_changes_with_seed():
    class Experience:
        pass

    experience = Experience()
    experience.dataset = [(torch.tensor([float(i)]), i) for i in range(10)]

    _, y1 = make_probe(experience, samples=5, seed=123)
    _, y2 = make_probe(experience, samples=5, seed=456)
    assert not torch.equal(y1, y2)
