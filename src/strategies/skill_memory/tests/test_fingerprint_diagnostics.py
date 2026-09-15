import importlib.util
import sys
import types
from pathlib import Path

import torch

root = Path(__file__).parents[1]


def _load_plugin():
    avalanche = types.ModuleType("avalanche")
    training = types.ModuleType("avalanche.training")
    plugins = types.ModuleType("avalanche.training.plugins")
    strategy_plugin = types.ModuleType("avalanche.training.plugins.strategy_plugin")

    class SupervisedPlugin:
        pass

    strategy_plugin.SupervisedPlugin = SupervisedPlugin
    plugins.strategy_plugin = strategy_plugin
    training.plugins = plugins
    avalanche.training = training

    models = types.ModuleType("avalanche.models")
    dynamic = types.ModuleType("avalanche.models.dynamic_modules")

    class IncrementalClassifier:
        pass

    dynamic.IncrementalClassifier = IncrementalClassifier
    dynamic.avalanche_model_adaptation = lambda model, experience: None
    models.dynamic_modules = dynamic
    avalanche.models = models
    sys.modules.update(
        {
            "avalanche": avalanche,
            "avalanche.training": training,
            "avalanche.training.plugins": plugins,
            "avalanche.training.plugins.strategy_plugin": strategy_plugin,
            "avalanche.models": models,
            "avalanche.models.dynamic_modules": dynamic,
        }
    )

    package = types.ModuleType("fingerprint_test_package")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package

    for name in (
        "behavior",
        "skill_registry",
        "probing",
        "decision",
        "training",
        "skill_memory_plugin",
        "persistent_skill_memory_plugin",
    ):
        module_name = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            root / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package.__name__}.persistent_skill_memory_plugin"]


def _record(mod, class_id, skill_id, values):
    return mod.ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=torch.ones(1, 2),
        output_class_ids=tuple(sorted(values)),
        reference_output=torch.tensor([values[c] for c in sorted(values)]),
        reference_summary=torch.ones(4),
    )


def test_fingerprint_route_records_reconstructable_per_sample_decisions():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    plugin.memory.store(1, {"slot": torch.tensor([1.0])})

    registry = sys.modules["fingerprint_test_package.skill_registry"]
    for class_id, skill_id, values in (
        (10, 0, {10: 1.0, 20: 0.0}),
        (20, 1, {10: 0.0, 20: 1.0}),
    ):
        plugin.class_map.record(
            registry.ClassRecord(
                experience_index=0,
                class_id=class_id,
                decision=plugin.SCRATCH,
                skill=skill_id,
            )
        )
        plugin.behavior.put(_record(mod, class_id, skill_id, values))

    strategy = types.SimpleNamespace(model=object())

    def fake_predict(model, state, x):
        del model
        slot = int(state["slot"].item())
        logits = torch.zeros(x.shape[0], 21)
        if slot == 0:
            logits[0, 10] = 9.0
        else:
            logits[1, 20] = 9.0
        return logits

    mod.predict_logits = fake_predict
    plugin._fingerprint_route(
        strategy,
        torch.tensor([[0.0], [1.0]]),
        [0, 1],
    )

    assert len(plugin.last_fingerprint_routes) == 2
    assert [item["skill"] for item in plugin.last_fingerprint_routes] == [0, 1]
    assert [item["class"] for item in plugin.last_fingerprint_routes] == [10, 20]
    for item in plugin.last_fingerprint_routes:
        assert item["sample_index"] in (0, 1)
        assert item["score"] >= item["second_score"]
        assert item["gap"] >= 0.0
        assert item["best_probability"] >= item["second_probability"]
        assert item["confidence_gap"] >= 0.0
        assert set(item["probabilities"]) == {0, 1}
        assert abs(sum(item["probabilities"].values()) - 1.0) < 1e-6
        assert item["best_probability"] == max(item["probabilities"].values())
        assert item["second_probability"] == min(item["probabilities"].values())
