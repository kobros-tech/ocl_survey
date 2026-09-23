import torch
from avalanche.models.dynamic_modules import IncrementalClassifier

from skill_memory.evaluation import behavior


def _record(class_id=42, skill_id=3, version=0):
    return behavior.ClassBehaviorRecord(
        class_id=class_id,
        skill_id=skill_id,
        version=version,
        reference_inputs=torch.ones(3, 2),
        reference_y=torch.tensor([True, True, True]),
    )


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = IncrementalClassifier(2, initial_out_features=3)

    def forward(self, x):
        return self.classifier(x)


def test_reverse_engineer_y_is_binary_and_uses_global_class_id():
    logits = torch.zeros(3, 100)
    logits[:, 42] = 10.0
    logits[1, 42] = -1.0
    logits[1, 87] = 20.0

    predicted = behavior.reverse_engineer_y(logits, 42)

    assert predicted.dtype == torch.bool
    assert predicted.tolist() == [True, False, True]


def test_reverse_engineer_y_is_not_multiclass_argmax():
    logits = torch.tensor(
        [
            [3.0, 1.0, -1.0],
            [4.0, 5.0, -1.0],
        ]
    )

    class_zero = behavior.reverse_engineer_y(logits, 0)
    class_one = behavior.reverse_engineer_y(logits, 1)

    assert class_zero.tolist() == [True, True]
    assert class_one.tolist() == [True, True]


def test_reverse_engineer_scores_ignore_inactive_classifier_units():
    model = _ToyModel()
    with torch.no_grad():
        model.classifier.classifier.weight.zero_()
        model.classifier.classifier.bias.zero_()
        model.classifier.classifier.weight[0, 0] = 100.0
        model.classifier.classifier.weight[1, 0] = 1.0
        model.classifier.active_units[:] = torch.tensor([0, 1, 0])

    x = torch.tensor([[1.0, 0.0]])
    actual = model(x).argmax(dim=-1)
    reconstructed = behavior.reverse_engineer_scores_from_weights(model, x).argmax(
        dim=-1
    )

    assert actual.tolist() == [1]
    assert reconstructed.tolist() == [1]


def test_known_class_prediction_is_compared_with_true_expected_y():
    predicted = torch.tensor([True, True, False, True])

    result = behavior.compare_binary_behavior(predicted, expected_y=True)

    assert result["expected_y"] is True
    assert result["correct"].tolist() == [True, True, False, True]
    assert result["accuracy"] == 0.75
    assert result["all_correct"] is False


def test_exact_anonymous_behavior_is_identified():
    predicted = torch.tensor([True, True, True])

    assert behavior.identify_binary_behavior(predicted, expected_y=True)


def test_nonmatching_anonymous_behavior_fails_identification():
    predicted = torch.tensor([True, False, True])

    assert not behavior.identify_binary_behavior(predicted, expected_y=True)


def test_cache_invalidates_every_class_of_changed_skill():
    cache = behavior.BehaviorFingerprintCache()
    cache.put(_record(class_id=42, skill_id=3))
    cache.put(_record(class_id=87, skill_id=3))
    cache.put(_record(class_id=12, skill_id=4))

    version = cache.bump_skill(3)

    assert version == 1
    assert cache.get(42, 3) is None
    assert cache.get(87, 3) is None
    assert cache.get(12, 4) is not None
    assert torch.equal(
        cache.all_records_for_skill(3)[0].reference_inputs,
        torch.ones(3, 2),
    )


def test_checkpoint_roundtrip_preserves_binary_fingerprint():
    cache = behavior.BehaviorFingerprintCache()
    record = _record(version=2)
    cache.put(record)

    restored = behavior.BehaviorFingerprintCache()
    restored.load_state_dict(cache.state_dict())
    result = restored.get(42, 3)

    assert result is not None
    assert result.version == 2
    assert result.expected_y is True
    assert torch.equal(result.reference_y, record.reference_y)
    assert torch.equal(result.reference_inputs, record.reference_inputs)


def test_legacy_checkpoint_without_behavior_state_loads():
    cache = behavior.BehaviorFingerprintCache()

    cache.load_state_dict({})

    state = cache.state_dict()
    assert state["skill_versions"] == {}
    assert state["skill_states"] == {}
    assert state["skill_state_versions"] == {}
    assert state["records"] == []
