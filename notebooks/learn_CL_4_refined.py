# %% [markdown]
# # Skill-memory continual learning (refined)
#
# This is the same demo as `learn_CL_3.ipynb`, refactored to call into the
# `skill_memory` package instead of redefining the model/strategy inline.
# It assumes `train_stream`, `test_stream`, and `device` already exist from
# your Avalanche benchmark setup earlier in the notebook -- paste those
# setup cells above this one unchanged.
#
# Changes vs. the original prototype:
# - old-data probe now actually gates reuse decisions (forgetting guard)
# - candidate clustering requires an absolute floor, not just relative gaps
#   (fixes the degenerate "always top-1" behavior at n=2 skills)
# - probes pull several batches instead of one batch of 10 (lower variance,
#   seedable via `SkillMemoryConfig.seed` for reproducible runs)
# - optional replay-on-reuse is available via config, off by default so you
#   can A/B it against the no-replay version

# %%
import numpy as np
import torch

from skill_memory import (
    SkillMemoryConfig,
    SkillClassifierBank,
    SkillMemoryStrategy,
    evaluate_seen_experiences,
    compute_cl_metrics,
)

# %%
config = SkillMemoryConfig(
    n_skills=10,
    input_dim=784,
    num_classes=10,
    batch_size=64,
    epochs_per_experience=1,
    learning_rate=0.01,
    probe_batch_size=10,
    probe_batches=5,       # was implicitly 1 batch of 10; now 5 batches (~50 samples)
    forgetting_margin=0.05,
    replay_old_during_reuse=False,  # flip to True to A/B against replay
    replay_batches_per_epoch=1,
    verbose=True,
    seed=0,                # set for reproducible probe sampling across runs
    device=str(device),
)

model = SkillClassifierBank(config).to(config.device)

optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

strategy = SkillMemoryStrategy(model, optimizer, criterion, config)

# %%
accuracy_history = []

for t, train_exp in enumerate(train_stream):
    strategy.train_experience(train_exp)

    current_accuracies = evaluate_seen_experiences(strategy, test_stream, t)
    accuracy_history.append(current_accuracies)

    print(
        f"Experience {t}: "
        f"trained on {sorted(train_exp.classes_in_this_experience)}, "
        f"mean seen accuracy = {np.mean(current_accuracies):.3f}"
    )

# %%
accuracy_curve, forgetting_curve = compute_cl_metrics(accuracy_history)

print("Accuracy:", np.round(accuracy_curve, 3))
print("Forgetting:", np.round(forgetting_curve, 3))

# %% [markdown]
# ## Suggested next experiment
#
# Run this cell block twice -- once with `replay_old_during_reuse=False`
# and once with `True` -- and compare the forgetting curves. That isolates
# exactly what the forgetting guard buys you (safer skill *selection*) vs.
# what replay buys you (less forgetting *during* training on a reused
# skill), which are two different mechanisms this version now separates
# cleanly.
