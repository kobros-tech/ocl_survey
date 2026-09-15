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

    package = types.ModuleType("persistent_test_package")
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


def test_changed_skill_collection_respects_mutability():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)
    plugin.last_class_decisions = {
        0: {
            0: {"decision": plugin.REUSE, "skill": 2},
            1: {"decision": plugin.SCRATCH, "skill": 3},
        }
    }

    plugin.reuse_is_mutable = True
    assert plugin._collect_changed_skills(0) == {2, 3}

    plugin.reuse_is_mutable = False
    assert plugin._collect_changed_skills(0) == {3}


def test_refresh_collects_decisions_after_training_hook():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)
    plugin._current_training_experience_index = 0
    plugin.last_class_decisions = {0: {7: {"decision": plugin.SCRATCH, "skill": 2}}}
    plugin.class_map.record(
        sys.modules["persistent_test_package.skill_registry"].ClassRecord(
            experience_index=0,
            class_id=7,
            decision=plugin.SCRATCH,
            skill=2,
        )
    )

    refreshed = []
    plugin._capture_new_class_inputs = lambda *args: None
    plugin._refresh_skill = lambda strategy, skill_id, experience: refreshed.append(
        skill_id
    )
    plugin._is_last_subexp = lambda experience: True
    mod.SkillMemoryPlugin.after_training_exp = lambda self, strategy, **kwargs: None

    experience = types.SimpleNamespace()
    strategy = types.SimpleNamespace(experience=experience)
    plugin.after_training_exp(strategy)

    assert refreshed == [2]
    assert plugin.behavior.skill_version(2) == 1


def test_new_class_reference_inputs_survive_multiple_subexperiences():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)
    plugin._current_training_experience_index = 0
    plugin.last_class_decisions = {0: {7: {"decision": plugin.SCRATCH, "skill": 2}}}

    captured = torch.tensor([[7.0, 8.0]])
    mod.probe_class = lambda *args: (captured, torch.tensor([7]))
    plugin._capture_new_class_inputs(types.SimpleNamespace(), 0)

    assert torch.equal(plugin._pending_reference_inputs[7], captured)


def test_multiclass_skill_routing_matches_persistent_class_behavior():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)

    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    plugin.memory.store(1, {"slot": torch.tensor([1.0])})

    registry = sys.modules["persistent_test_package.skill_registry"]
    for class_id, values in (
        (0, {0: 1.0, 1: 0.0}),
        (1, {0: 0.0, 1: 1.0}),
        (2, {2: 1.0, 3: 0.0}),
        (3, {2: 0.0, 3: 1.0}),
    ):
        skill_id = 0 if class_id < 2 else 1
        plugin.class_map.record(
            registry.ClassRecord(
                experience_index=0,
                class_id=class_id,
                decision=plugin.SCRATCH,
                skill=skill_id,
            )
        )
        plugin.behavior.put(_record(mod, class_id, skill_id, values))

    class Model:
        pass

    strategy = types.SimpleNamespace(model=Model())

    def fake_predict(model, state, x):
        del model
        slot = int(state["slot"].item())
        rows = []
        for value in x[:, 0].tolist():
            if slot == 0 and value < 2:
                rows.append(
                    [8.0, 0.0, 0.0, 0.0] if value == 0 else [0.0, 8.0, 0.0, 0.0]
                )
            elif slot == 1:
                rows.append(
                    [0.0, 0.0, 8.0, 0.0] if value == 2 else [0.0, 0.0, 0.0, 8.0]
                )
            else:
                rows.append([0.0, 0.0, 0.0, 0.0])
        return torch.tensor(rows)

    mod.predict_logits = fake_predict
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    chosen, probabilities, classes = plugin._fingerprint_route(
        strategy,
        x,
        [0, 1],
    )

    assert chosen.tolist() == [0, 0, 1, 1]
    assert classes == [0, 1, 2, 3]
    assert probabilities.shape == (2, 4)


def test_legacy_behavior_checkpoint_is_accepted():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)
    plugin.load_state_dict({})
    assert not plugin._behavior_initialized
