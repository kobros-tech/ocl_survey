"""Global synchronization of reverse-engineering fingerprints."""

from __future__ import annotations

from typing import Any


def refresh_all_fingerprints(plugin: Any, strategy: Any) -> dict[str, int]:
    """Rebuild every persisted class fingerprint from current skill weights.

    Reference inputs are intentionally retained. Only the reverse-engineered
    behavior and continuous statistics are recalculated, using the current
    canonical skill state for every mastered class. This keeps the routing
    representation synchronized with the model after a complete logical
    training phase instead of refreshing only the experience that changed.
    """
    records = list(plugin.behavior.state_dict().get("records", []))
    refreshed = 0
    skipped = 0
    touched_skills: set[int] = set()

    for record_state in records:
        class_id = int(record_state["class_id"])
        skill_id = plugin.class_map.find_skill_for_class_anywhere(class_id)
        if skill_id is None:
            skipped += 1
            continue
        skill_id = int(skill_id)
        state_dict = plugin.memory.state(skill_id)
        version = plugin.behavior.skill_version(skill_id)
        plugin.behavior.put_skill_state(skill_id, version, state_dict)
        reference_inputs = record_state["reference_inputs"].detach().cpu().clone()
        refreshed_record = plugin._build_record(
            strategy,
            skill_id,
            class_id,
            reference_inputs,
            version,
            state_dict,
        )
        plugin.behavior.put(refreshed_record)
        refreshed += 1
        touched_skills.add(skill_id)

    plugin._behavior_initialized = bool(
        plugin.behavior.state_dict().get("records")
    )
    return {
        "records_seen": len(records),
        "records_refreshed": refreshed,
        "records_skipped": skipped,
        "skills_refreshed": len(touched_skills),
    }
