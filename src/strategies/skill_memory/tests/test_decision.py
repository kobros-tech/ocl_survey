from skill_memory.cl.decision import find_best_skill


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
