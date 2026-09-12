from pathlib import Path
import importlib.util
import sys
import types

# Minimal Avalanche stubs so decision.py/probing.py can be imported in a
# lightweight environment without installing Avalanche.
avalanche = types.ModuleType("avalanche")
models = types.ModuleType("avalanche.models")
dynamic = types.ModuleType("avalanche.models.dynamic_modules")
class IncrementalClassifier:  # pragma: no cover - only used for isinstance
    pass
dynamic.IncrementalClassifier = IncrementalClassifier
dynamic.avalanche_model_adaptation = lambda model, experience: None
models.dynamic_modules = dynamic
avalanche.models = models
training = types.ModuleType("avalanche.training")
plugins = types.ModuleType("avalanche.training.plugins")
sp = types.ModuleType("avalanche.training.plugins.strategy_plugin")
class SupervisedPlugin:  # pragma: no cover
    pass
sp.SupervisedPlugin = SupervisedPlugin
plugins.strategy_plugin = sp
training.plugins = plugins
avalanche.training = training
sys.modules.update({
    "avalanche": avalanche,
    "avalanche.models": models,
    "avalanche.models.dynamic_modules": dynamic,
    "avalanche.training": training,
    "avalanche.training.plugins": plugins,
    "avalanche.training.plugins.strategy_plugin": sp,
})

root = Path(__file__).parents[1]
pkg = types.ModuleType("skillpkg")
pkg.__path__ = [str(root)]
sys.modules["skillpkg"] = pkg

for name in ("skill_registry", "probing", "decision"):
    path = root / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"skillpkg.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

find_best_skill = sys.modules["skillpkg.decision"].find_best_skill


def test_reuse_requires_worst_old_class_to_be_safe():
    results = [
        {
            "skill": 0,
            "old_accuracy": 0.95,  # aggregate/worst old-class accuracy
            "chance": 0.10,
            "new_score": 0.95,
            "new_accuracy": 0.95,
        },
        {
            "skill": 1,
            "old_accuracy": 0.10,
            "chance": 0.10,
            "new_score": 0.99,
            "new_accuracy": 0.99,
        },
    ]
    best = find_best_skill(results, forgetting_margin=0.05, score_floor=0.9)
    assert best is not None
    assert best["skill"] == 0
