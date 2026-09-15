"""Skill Memory extension using persistent class behavior fingerprints."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor

from .behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    extract_reference_behavior,
    probe_behavior_fingerprint,
)
from .probing import _normalize_routing_scores, predict_logits, probe_class
from .skill_memory_plugin import SkillMemoryPlugin


class PersistentFingerprintSkillMemoryPlugin(SkillMemoryPlugin):
    """Skill Memory with persistent anonymous class-behavior routing.

    Reference behavior is generated after training and reused during every
    evaluation pass. Mutable REUSE changes create a new skill generation and
    refresh every class mastered by that skill. SCRATCH creates references for
    the new skill. Unchanged skills keep their existing references.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior = BehaviorFingerprintCache()
        self._behavior_initialized = False
        self._pending_reference_inputs: dict[int, Tensor] = {}
        self.last_fingerprint_routes: list[dict] = []

    def _build_record(
        self,
        strategy,
        skill_id: int,
        class_id: int,
        x: Tensor,
        version: int,
    ) -> ClassBehaviorRecord:
        model = deepcopy(strategy.model)
        logits = predict_logits(model, self.memory.state(skill_id), x)
        output_class_ids = list(range(logits.shape[-1]))
        output, summary, output_class_ids = extract_reference_behavior(
            logits,
            output_class_ids,
            class_id,
        )
        return ClassBehaviorRecord(
            class_id=class_id,
            skill_id=skill_id,
            version=version,
            reference_inputs=x.detach().cpu().clone(),
            output_class_ids=output_class_ids,
            reference_output=output,
            reference_summary=summary,
        )

    def _capture_new_class_inputs(self, experience, experience_index: int) -> None:
        """Keep probe inputs for newly introduced classes until final refresh."""
        decisions = self.last_class_decisions.get(experience_index, {})
        for class_id, item in decisions.items():
            skill_id = item.get("skill")
            if skill_id is None:
                continue
            if self.behavior.get(class_id, int(skill_id)) is not None:
                continue
            if class_id in self._pending_reference_inputs:
                continue
            x, _ = probe_class(
                experience,
                class_id,
                self.probe_batch_size,
                self.probe_batches,
                self.probe_seed,
            )
            self._pending_reference_inputs[class_id] = x.detach().cpu().clone()

    def _refresh_skill(self, strategy, skill_id: int, experience) -> None:
        """Refresh all class references for one current skill generation."""
        version = self.behavior.skill_version(skill_id)
        classes = sorted(self.class_map.classes_for_skill(skill_id))
        if not classes:
            return

        existing = {
            record.class_id: record
            for record in self.behavior.all_records_for_skill(skill_id)
        }
        for class_id in classes:
            record = existing.get(class_id)
            if record is None:
                x = self._pending_reference_inputs.get(class_id)
                if x is None:
                    x, _ = probe_class(
                        experience,
                        class_id,
                        self.probe_batch_size,
                        self.probe_batches,
                        self.probe_seed,
                    )
            else:
                x = record.reference_inputs

            self.behavior.put(
                self._build_record(strategy, skill_id, class_id, x, version)
            )

    def _collect_changed_skills(self, experience_index: int) -> set[int]:
        decisions = self.last_class_decisions.get(experience_index, {})
        changed: set[int] = set()
        for item in decisions.values():
            decision = item.get("decision")
            skill_id = item.get("skill")
            if skill_id is None:
                continue
            if decision == self.SCRATCH:
                changed.add(int(skill_id))
            elif decision == self.REUSE and self.reuse_is_mutable:
                changed.add(int(skill_id))
        return changed

    def after_training_exp(self, strategy, **kwargs) -> None:
        """Refresh changed skills once all sub-experiences are complete."""
        experience = strategy.experience
        experience_index = self._current_training_experience_index
        is_last = self._is_last_subexp(experience)

        if experience_index is not None:
            self._capture_new_class_inputs(experience, experience_index)

        changed = (
            self._collect_changed_skills(experience_index)
            if is_last and experience_index is not None
            else set()
        )

        super().after_training_exp(strategy, **kwargs)
        if experience_index is None or not is_last:
            return

        for skill_id in sorted(changed):
            self.behavior.bump_skill(skill_id)
            self._refresh_skill(strategy, skill_id, experience)

        self._pending_reference_inputs.clear()
        self._behavior_initialized = bool(self.behavior.state_dict()["records"])

    def _fingerprint_route(
        self,
        strategy,
        x: Tensor,
        slot_ids: list[int],
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Match each probe to a canonical class, then its canonical skill."""
        probe_model = deepcopy(strategy.model)
        raw_logits = {
            slot: predict_logits(probe_model, self.memory.state(slot), x)
            for slot in slot_ids
        }
        slot_to_row = {slot: row for row, slot in enumerate(slot_ids)}

        records = [
            record
            for slot in slot_ids
            for record in self.behavior.records_for_skill(slot)
        ]
        if not records:
            self.last_fingerprint_routes = []
            chosen = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            probabilities = torch.full(
                (len(slot_ids), x.shape[0]),
                1.0 / max(len(slot_ids), 1),
                device=x.device,
            )
            return chosen, probabilities, [-1] * x.shape[0]

        class_scores: list[Tensor] = []
        class_ids: list[int] = []
        class_skills: list[int] = []
        for record in records:
            similarity, _ = probe_behavior_fingerprint(
                raw_logits[record.skill_id],
                record.output_class_ids,
                record.reference_output,
                record.reference_summary,
            )
            class_scores.append(similarity)
            class_ids.append(record.class_id)
            class_skills.append(record.skill_id)

        stacked = torch.stack(class_scores, dim=0)
        best_scores, best_indices = stacked.max(dim=0)
        chosen_classes = [class_ids[index] for index in best_indices.cpu().tolist()]
        chosen_skills = [class_skills[index] for index in best_indices.cpu().tolist()]

        # Fingerprint similarity is a signed similarity score, not a
        # probability. Convert it to bounded routing evidence first, then use
        # the same normalization contract as input-only skill routing.
        skill_scores = torch.zeros(
            (len(slot_ids), x.shape[0]),
            device=x.device,
        )
        for score, skill in zip(class_scores, class_skills, strict=False):
            evidence = ((score + 1.0) / 2.0).clamp(0.0, 1.0)
            skill_scores[slot_to_row[skill]] = torch.maximum(
                skill_scores[slot_to_row[skill]], evidence
            )

        probabilities = _normalize_routing_scores(skill_scores, temperature=1.0)
        chosen = torch.tensor(
            [slot_to_row[skill] for skill in chosen_skills],
            dtype=torch.long,
            device=x.device,
        )

        self.last_fingerprint_routes = []
        for sample_index, skill in enumerate(chosen_skills):
            sample_class_scores = stacked[:, sample_index]
            if sample_class_scores.numel() > 1:
                top_scores = torch.topk(sample_class_scores, k=2).values
                second_score = float(top_scores[1].item())
            else:
                second_score = 0.0
            sample_probabilities = probabilities[:, sample_index]
            top_probabilities = torch.topk(
                sample_probabilities,
                k=min(2, sample_probabilities.numel()),
            ).values
            best_probability = float(top_probabilities[0].item())
            second_probability = (
                float(top_probabilities[1].item())
                if top_probabilities.numel() > 1
                else 0.0
            )
            self.last_fingerprint_routes.append(
                {
                    "sample_index": sample_index,
                    "skill": int(skill),
                    "class": int(chosen_classes[sample_index]),
                    "score": float(best_scores[sample_index].item()),
                    "second_score": second_score,
                    "gap": float(best_scores[sample_index].item()) - second_score,
                    "best_probability": best_probability,
                    "second_probability": second_probability,
                    "confidence_gap": best_probability - second_probability,
                    "probabilities": {
                        int(slot): float(probabilities[row, sample_index].item())
                        for row, slot in enumerate(slot_ids)
                    },
                }
            )

        return chosen, probabilities, chosen_classes

    def after_eval_forward(self, strategy, **kwargs) -> None:
        if not self._eval_active or self.eval_routing != "probe":
            return super().after_eval_forward(strategy, **kwargs)
        if len(self.memory) == 0 or not self._behavior_initialized:
            return super().after_eval_forward(strategy, **kwargs)

        x = strategy.mbatch[0]
        slot_ids = sorted(self.memory.slots())
        chosen, probabilities, class_matches = self._fingerprint_route(
            strategy,
            x,
            slot_ids,
        )

        output_model = deepcopy(strategy.model)
        per_skill_logits = [
            predict_logits(output_model, self.memory.state(slot), x)
            for slot in slot_ids
        ]
        output_dim = strategy.mb_output.shape[-1]
        padded = []
        for logits in per_skill_logits:
            result = logits.new_full((logits.shape[0], output_dim), -1e4)
            width = min(logits.shape[-1], output_dim)
            result[:, :width] = logits[:, :width]
            padded.append(result)

        strategy.mb_output = torch.stack(padded, dim=0)[
            chosen,
            torch.arange(x.shape[0], device=x.device),
        ]

        best = probabilities.max(dim=0).values
        self._log(
            "[FINGERPRINT routing] samples="
            f"{x.shape[0]} mean_probability={best.mean().item():.4f} "
            f"matched_classes={class_matches[:5]}"
        )

    def state_dict(self) -> dict:
        """Serialize behavior state; missing state remains backward compatible."""
        return {"behavior": self.behavior.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore behavior state from a current or legacy checkpoint."""
        behavior_state = state.get("behavior", {})
        self.behavior.load_state_dict(behavior_state)
        self._behavior_initialized = bool(behavior_state.get("records"))
