from skill_memory.cl.skill_memory_plugin import SkillMemoryPlugin


def test_probe_is_default_and_class_oracle_is_diagnostic():
    plugin = SkillMemoryPlugin(verbose=False)
    assert plugin.eval_routing == "probe"
    oracle = SkillMemoryPlugin(eval_routing="class_oracle", verbose=False)
    assert oracle.eval_routing == "class_oracle"
