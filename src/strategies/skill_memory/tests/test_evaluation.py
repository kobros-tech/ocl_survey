import torch

from skill_memory.evaluation import find_best_routing_skill, route_probe_logits


def test_route_probe_logits_selects_one_skill_per_sample_without_labels():
    logits_by_skill = [
        torch.tensor([[4.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [4.0, 1.0]]),
    ]

    chosen = route_probe_logits(
        logits_by_skill,
        [{"classifier.weight": torch.tensor([[2.0, 0.0]])}] * 2,
        [{0}, {0}],
    )

    assert chosen.tolist() == [0, 1]


def test_one_class_heads_use_raw_logit_confidence():
    result = find_best_routing_skill(
        [
            torch.tensor([[5.0], [1.0]]),
            torch.tensor([[1.0], [5.0]]),
        ],
        [{"classifier.weight": torch.tensor([[1.0]])}] * 2,
        [{0}, {0}],
    )

    assert result.skill_indices.tolist() == [0, 1]
    assert torch.all(result.confidence_gap > 0)


def test_routing_uses_owned_global_class_columns():
    class_a = 87
    class_b = 42
    logits_a = torch.zeros(2, 88)
    logits_b = torch.zeros(2, 88)
    logits_a[:, class_a] = torch.tensor([5.0, 1.0])
    logits_b[:, class_b] = torch.tensor([1.0, 5.0])

    chosen = route_probe_logits(
        [logits_a, logits_b],
        [{"classifier.weight": torch.tensor([[1.0]])}] * 2,
        [{class_a}, {class_b}],
    )

    assert chosen.tolist() == [0, 1]


def test_routing_raises_instead_of_silently_zeroing_a_skill_with_narrow_output():
    """A skill's *own* raw output must already have a column for every class
    it owns (Avalanche's IncrementalClassifier only ever grows a skill's own
    classifier to cover the classes that skill was actually trained on - see
    _routing_scores' docstring). If a skill's owned classes fall outside its
    own raw output width, class bookkeeping and the model have drifted apart
    and the skill would otherwise always score zero and never win routing -
    a routing failure that looks like the wrong bug. This must raise instead
    of silently returning a zero score for that skill.
    """
    # Skill owns global class 37 but its own raw output is only 5-wide.
    narrow_logits = torch.zeros(2, 5)
    wide_logits = torch.zeros(2, 40)
    wide_logits[:, 37] = 5.0

    try:
        route_probe_logits(
            [narrow_logits, wide_logits],
            [{"classifier.weight": torch.tensor([[1.0]])}] * 2,
            [{37}, {37}],
        )
    except RuntimeError as exc:
        assert "37" in str(exc)
        assert "drifted apart" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError for the narrow-output skill")
