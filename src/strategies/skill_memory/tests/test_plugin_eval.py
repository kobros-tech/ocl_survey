from pathlib import Path
import importlib.util
import sys
import types

import torch

# Lightweight Avalanche stubs.
avalanche = types.ModuleType("avalanche")
training = types.ModuleType("avalanche.training")
plugins = types.ModuleType("avalanche.training.plugins")
sp = types.ModuleType("avalanche.training.plugins.strategy_plugin")


class SupervisedPlugin:
    pass


sp.SupervisedPlugin = SupervisedPlugin
plugins.strategy_plugin = sp
training.plugins = plugins
avalanche.training = training
sys.modules.update(
    {
        "avalanche": avalanche,
        "avalanche.training": training,
        "avalanche.training.plugins": plugins,
        "avalanche.training.plugins.strategy_plugin": sp,
    }
)

# The plugin imports probing/decision/registry/training as package-relative
# modules. Load the package under an isolated name and provide the minimum
# Avalanche model symbols needed by probing.py.
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
        "avalanche.models": models,
        "avalanche.models.dynamic_modules": dynamic,
    }
)

root = Path(__file__).parents[1]
pkg = types.ModuleType("pluginpkg")
pkg.__path__ = [str(root)]
sys.modules["pluginpkg"] = pkg

for name in ("skill_registry", "probing", "decision", "training", "skill_memory_plugin"):
    path = root / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"pluginpkg.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

SkillMemoryPlugin = sys.modules["pluginpkg.skill_memory_plugin"].SkillMemoryPlugin


def test_probe_is_default_and_class_oracle_is_diagnostic():
    plugin = SkillMemoryPlugin(verbose=False)
    assert plugin.eval_routing == "probe"
    oracle = SkillMemoryPlugin(eval_routing="class_oracle", verbose=False)
    assert oracle.eval_routing == "class_oracle"
