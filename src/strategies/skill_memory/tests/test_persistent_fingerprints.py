import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]


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
    training.plugins = plugins
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

    package = types.ModuleType("binary_fingerprint_test")
    package.__path__ = [str(ROOT)]
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
            ROOT / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package.__name__}.persistent_skill_memory_plugin"]


def _record(mod, class_id, skill_id):
    return mod.ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=0,
        reference_inputs=torch.ones(2, 1),
        reference_y=torch.tensor([True, True]),
    )


def _registry_record(registry, plugin, class_id, skill_id):
    return registry.ClassRecord(
        experience_index=0,
        class_id=class_id,
        decision=plugin.SCRATCH,
        skill=skill_id,
    )


def test_binary_route_identifies_one_candidate_and_logs_evidence():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(
        verbose=False,
        reverse_engineer_y_fn=lambda logits, target: logits[:, target] > 0,
    )
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    plugin.memory.store(1, {"slot": torch.tensor([1.0])})

    registry = sys.modules["binary_fingerprint_test.skill_registry"]
    plugin.class_map.record(_registry_record(registry, plugin, 10, 0))
    plugin.class_map.record(_registry_record(registry, plugin, 20, 1))
    plugin.behavior.put(_record(mod, 10, 0))
    plugin.behavior.put(_record(mod, 20, 1))

    class Model:
        pass

    strategy = types.SimpleNamespace(model=Model())

    def fake_predict(model, state, x):
        del model
        logits = torch.full((x.shape[0], 21), -1.0)
        slot = int(state["slot"].item())
        for row, value in enumerate(x[:, 0].tolist()):
            if slot == 0 and value == 10:
                logits[row, 10] = 1.0
            if slot == 1 and value == 20:
                logits[row, 20] = 1.0
        return logits

    mod.predict_logits = fake_predict
    chosen, classes = plugin._fingerprint_route(
        strategy,
        torch.tensor([[10.0], [20.0]]),
        [0, 1],
    )

    assert chosen.tolist() == [0, 1]
    assert classes == [10, 20]
    assert [item["status"] for item in plugin.last_fingerprint_routes] == [
        "IDENTIFIED",
        "IDENTIFIED",
    ]
    for item in plugin.last_fingerprint_routes:
        assert len(item["candidates"]) == 2
        assert all(
            {"class", "skill", "predicted_y", "expected_y", "correct"} <= set(candidate)
            for candidate in item["candidates"]
        )


def test_binary_route_is_ambiguous_when_multiple_candidates_return_true():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(
        verbose=False,
        reverse_engineer_y_fn=lambda logits, target: torch.ones(
            logits.shape[0], dtype=torch.bool
        ),
    )
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    plugin.memory.store(1, {"slot": torch.tensor([1.0])})
    registry = sys.modules["binary_fingerprint_test.skill_registry"]
    plugin.class_map.record(_registry_record(registry, plugin, 10, 0))
    plugin.class_map.record(_registry_record(registry, plugin, 20, 1))
    plugin.behavior.put(_record(mod, 10, 0))
    plugin.behavior.put(_record(mod, 20, 1))

    class Model:
        pass

    mod.predict_logits = lambda model, state, x: torch.zeros(x.shape[0], 21)
    chosen, classes = plugin._fingerprint_route(
        types.SimpleNamespace(model=Model()),
        torch.tensor([[0.0]]),
        [0, 1],
    )

    assert chosen.tolist() == [-1]
    assert classes == [-1]
    assert plugin.last_fingerprint_routes[0]["status"] == "AMBIGUOUS"
    assert len(plugin.last_fingerprint_routes[0]["candidates"]) == 2


def test_binary_route_fails_when_no_candidate_returns_expected_y():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(
        verbose=False,
        reverse_engineer_y_fn=lambda logits, target: torch.zeros(
            logits.shape[0], dtype=torch.bool
        ),
    )
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    registry = sys.modules["binary_fingerprint_test.skill_registry"]
    plugin.class_map.record(_registry_record(registry, plugin, 10, 0))
    plugin.behavior.put(_record(mod, 10, 0))

    class Model:
        pass

    mod.predict_logits = lambda model, state, x: torch.zeros(x.shape[0], 11)
    chosen, classes = plugin._fingerprint_route(
        types.SimpleNamespace(model=Model()),
        torch.tensor([[0.0]]),
        [0],
    )

    assert chosen.tolist() == [-1]
    assert classes == [-1]
    assert plugin.last_fingerprint_routes[0]["status"] == "FAILED"


def test_multiclass_skill_keeps_class_identity_then_resolves_same_skill():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(
        verbose=False,
        reverse_engineer_y_fn=lambda logits, target: logits[:, target] > 0,
    )
    plugin.memory.store(0, {"slot": torch.tensor([0.0])})
    registry = sys.modules["binary_fingerprint_test.skill_registry"]
    for class_id in (10, 20):
        plugin.class_map.record(_registry_record(registry, plugin, class_id, 0))
        plugin.behavior.put(_record(mod, class_id, 0))

    class Model:
        pass

    def fake_predict(model, state, x):
        del model, state
        logits = torch.full((x.shape[0], 21), -1.0)
        for row, value in enumerate(x[:, 0].tolist()):
            logits[row, int(value)] = 1.0
        return logits

    mod.predict_logits = fake_predict
    chosen, classes = plugin._fingerprint_route(
        types.SimpleNamespace(model=Model()),
        torch.tensor([[10.0], [20.0]]),
        [0],
    )

    assert chosen.tolist() == [0, 0]
    assert classes == [10, 20]


def test_eval_output_uses_skill_id_not_slot_position():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(
        verbose=False,
        reverse_engineer_y_fn=lambda logits, target: logits[:, target] > 0,
    )
    plugin.memory.store(2, {"slot": torch.tensor([2.0])})
    plugin.memory.store(7, {"slot": torch.tensor([7.0])})
    registry = sys.modules["binary_fingerprint_test.skill_registry"]
    plugin.class_map.record(_registry_record(registry, plugin, 10, 2))
    plugin.class_map.record(_registry_record(registry, plugin, 20, 7))
    plugin.behavior.put(_record(mod, 10, 2))
    plugin.behavior.put(_record(mod, 20, 7))
    plugin._behavior_initialized = True

    class Model:
        pass

    def fake_predict(model, state, x):
        del model
        slot = int(state["slot"].item())
        logits = torch.full((x.shape[0], 21), -1.0)
        for row, value in enumerate(x[:, 0].tolist()):
            target = 10 if value == 10 else 20
            if (slot, target) in ((2, 10), (7, 20)):
                logits[row, target] = 5.0
            else:
                logits[row, target] = -2.0
        return logits

    mod.predict_logits = fake_predict
    strategy = types.SimpleNamespace(
        model=Model(),
        mbatch=(torch.tensor([[10.0], [20.0]]), torch.tensor([10, 20])),
        mb_output=torch.zeros(2, 21),
    )
    plugin._eval_active = True
    plugin.eval_routing = "probe"

    plugin.after_eval_forward(strategy)

    assert strategy.mb_output[0, 10].item() == 5.0
    assert strategy.mb_output[1, 20].item() == 5.0


def test_immutable_reuse_does_not_mark_skill_changed():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(
        verbose=False,
        reuse_is_mutable=False,
    )
    plugin.last_class_decisions = {
        0: {
            10: {"decision": plugin.REUSE, "skill": 0},
            20: {"decision": plugin.SCRATCH, "skill": 1},
        }
    }

    assert plugin._collect_changed_skills(0) == {1}


def test_legacy_checkpoint_loads_without_behavior_records():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(verbose=False)

    plugin.load_state_dict({})

    assert plugin._behavior_initialized is False


def test_skill_generation_state_is_persisted_and_reused():
    mod = _load_plugin()
    cache = mod.BehaviorFingerprintCache()
    state = {"weight": torch.tensor([1.0, 2.0])}

    cache.put_skill_state(3, 0, state)
    state["weight"][0] = 99.0

    frozen = cache.skill_state(3, 0)
    assert frozen is not None
    assert torch.equal(frozen["weight"], torch.tensor([1.0, 2.0]))

    restored = mod.BehaviorFingerprintCache()
    restored.load_state_dict(cache.state_dict())
    restored_state = restored.skill_state(3, 0)
    assert restored_state is not None
    assert torch.equal(restored_state["weight"], torch.tensor([1.0, 2.0]))


def test_route_uses_frozen_generation_instead_of_current_skill_state():
    mod = _load_plugin()
    plugin = mod.PersistentFingerprintSkillMemoryPlugin(
        verbose=False,
        reverse_engineer_y_fn=lambda logits, target: logits[:, target] > 0,
    )
    plugin.memory.store(0, {"slot": torch.tensor([99.0])})
    registry = sys.modules["binary_fingerprint_test.skill_registry"]
    plugin.class_map.record(_registry_record(registry, plugin, 10, 0))
    plugin.behavior.put(_record(mod, 10, 0))
    plugin.behavior.put_skill_state(0, 0, {"slot": torch.tensor([10.0])})

    class Model:
        pass

    seen_slots = []

    def fake_predict(model, state, x):
        del model
        seen_slots.append(int(state["slot"].item()))
        logits = torch.full((x.shape[0], 11), -1.0)
        if int(state["slot"].item()) == 10:
            logits[:, 10] = 1.0
        return logits

    mod.predict_logits = fake_predict
    chosen, classes = plugin._fingerprint_route(
        types.SimpleNamespace(model=Model()),
        torch.tensor([[0.0]]),
        [0],
    )

    assert chosen.tolist() == [0]
    assert classes == [10]
    assert seen_slots == [10]
