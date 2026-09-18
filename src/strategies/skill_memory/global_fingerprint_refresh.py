"""Global synchronization of reverse-engineering fingerprints."""

from __future__ import annotations

from typing import Any


def refresh_all_fingerprints(plugin: Any, strategy: Any) -> dict[str, int]:
    """Rebuild persisted class fingerprints from current canonical skill weights.

    A record only counts as "refreshed" once it has actually been written
    back via ``plugin.behavior.put``. A class whose skill mapping still
    exists but whose rebuilt record came back invalid (``None``) is counted
    under ``records_invalid`` instead, so a stale fingerprint is never
    reported as if it had been updated.
    """
    records = list(plugin.behavior.state_dict().get("records", []))
    attempted = 0
    written = 0
    invalid = 0
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
        attempted += 1
        if refreshed_record is None:
            invalid += 1
            continue
        plugin.behavior.put(refreshed_record)
        written += 1
        touched_skills.add(skill_id)

    plugin._behavior_initialized = bool(plugin.behavior.state_dict().get("records"))
    return {
        "records_seen": len(records),
        "records_attempted": attempted,
        "records_refreshed": written,
        "records_invalid": invalid,
        "records_skipped": skipped,
        "skills_refreshed": len(touched_skills),
    }
