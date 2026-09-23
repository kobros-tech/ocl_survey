"""Compatibility entry point for persistent anonymous routing."""

from __future__ import annotations

from ..cl.persistent_skill_memory_plugin import (
    PersistentFingerprintSkillMemoryPlugin as _BaseFingerprintPlugin,
)
from .diagnostics import class_index_alignment_report, routing_rank_diagnostics


class PersistentFingerprintSkillMemoryPlugin(_BaseFingerprintPlugin):
    """Persistent router with lightweight defaults and optional diagnostics.

    ``diagnose=False`` (the default) disables route-history retention *and*
    the underlying per-candidate diagnostic breakdown in `_route` itself, so
    the listwise routing forward pass skips the extra per-sample,
    per-candidate dict construction and GPU->CPU syncs entirely rather than
    building it and discarding it. Set ``diagnose=True`` when detailed
    routing records (`last_routing_diagnostics`) are needed - this also
    computes `last_alignment_report` once per completed training experience
    (see `class_index_alignment_report`), which is the concrete, runnable
    check for whether a skill's owned global class ids actually fit its own
    classifier's output space on this benchmark.
    """

    def __init__(
        self,
        *args,
        reverse_epochs: int = 60,
        reverse_batch_size: int = 256,
        diagnose: bool = False,
        **kwargs,
    ) -> None:
        """Wrap the base plugin, defaulting diagnostics off for routing speed."""
        self.diagnose = bool(diagnose)
        self.last_routing_diagnostics: dict = {}
        self.last_alignment_report: dict = {}
        kwargs.setdefault("record_candidate_diagnostics", self.diagnose)
        super().__init__(
            *args,
            reverse_epochs=reverse_epochs,
            reverse_batch_size=reverse_batch_size,
            **kwargs,
        )

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Fit the router, then refresh `last_alignment_report` if diagnosing."""
        super().after_training_exp(strategy, **kwargs)
        if not self.diagnose:
            self.last_alignment_report = {}
            return
        slot_ids = sorted(self.memory.slots())
        if not slot_ids:
            self.last_alignment_report = {}
            return
        self.last_alignment_report = class_index_alignment_report(
            strategy.model,
            states=[self.memory.state(slot) for slot in slot_ids],
            slot_ids=slot_ids,
            owned_classes_by_slot=[
                self.class_map.classes_for_skill(slot) for slot in slot_ids
            ],
        )

    def after_eval_forward(self, strategy, **kwargs) -> None:
        """Route as usual, then refresh or discard `last_routing_diagnostics`."""
        super().after_eval_forward(strategy, **kwargs)
        if self.diagnose:
            self.last_routing_diagnostics = routing_rank_diagnostics(
                self.fingerprint_route_history
            )
        else:
            self.last_routing_diagnostics = {}
            self.fingerprint_route_history.clear()


__all__ = ["PersistentFingerprintSkillMemoryPlugin"]
