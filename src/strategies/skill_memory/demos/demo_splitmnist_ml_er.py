"""SplitMNIST Skill Memory experiment with anonymous ML evaluation.

The public SkillMemoryStrategy owns the complete experiment lifecycle:

* Skill Memory training
* frozen per-class evaluation memory
* independent ML evaluator training
* anonymous x -> y evaluation through Avalanche's normal eval() lifecycle
* accuracy/loss tracking
* forgetting metrics

The demo only configures SplitMNIST and the strategy, then reports results.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from torch import nn

from skill_memory import SkillMemoryStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Skill Memory on SplitMNIST."
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument(
        "--eval-memory-per-class",
        type=int,
        default=20,
        help="Frozen evaluation examples retained per class.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=1,
        help="Number of epochs used by the independent ML evaluator.",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument(
        "--skill-eval-routing",
        choices=("oracle", "probe", "both", "none"),
        default="none",
        help=(
            "Optional direct Skill Memory diagnostic. 'oracle' uses the "
            "true class-to-skill mapping; 'probe' uses anonymous routing; "
            "'both' runs both; 'none' disables direct Skill Memory "
            "evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = SplitMNIST(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

    print("=== SplitMNIST Skill Memory experiment ===")
    print("Training method: Skill Memory")
    print("Evaluation method: independent anonymous ML evaluator")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    print(
        "Evaluation memory per class:",
        args.eval_memory_per_class,
    )

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    # ------------------------------------------------------------------
    # Main Skill Memory model.
    # ------------------------------------------------------------------

    model = SimpleMLP(
        num_classes=10,
    ).to(device)

    # ------------------------------------------------------------------
    # SkillMemoryStrategy owns:
    #
    #   1. Skill Memory training
    #   2. evaluation-memory retention
    #   3. independent ML evaluator
    #   4. normal Avalanche evaluation lifecycle
    #
    # The evaluator receives only x at evaluation time and predicts y.
    # No experience ID, task label, or skill ID is supplied.
    # ------------------------------------------------------------------

    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
        ),
        criterion=nn.CrossEntropyLoss(),
        max_skills=args.max_skills,
        train_mb_size=args.batch_size,
        train_epochs=args.train_epochs,
        eval_mb_size=args.eval_batch_size,
        evaluator_model_factory=lambda: SimpleMLP(
            num_classes=10,
            input_size=28 * 28,
            hidden_size=512,
            hidden_layers=1,
            drop_rate=0.5,
        ),
        eval_memory_per_class=args.eval_memory_per_class,
        eval_epochs=args.eval_epochs,
        eval_learning_rate=args.eval_learning_rate,
        skill_eval_routing=args.skill_eval_routing,
        probe_seed=args.seed,
        device=device,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Complete Avalanche lifecycle.
    #
    # Training:
    #     strategy.train(experience)
    #
    # Evaluation:
    #     strategy.eval(benchmark.test_stream)
    #
    # The ML evaluator is invoked automatically by its Avalanche plugin
    # during strategy.eval().
    # ------------------------------------------------------------------

    for experience_index, experience in enumerate(benchmark.train_stream):
        print()
        print(f"========== Training experience {experience_index} ==========")
        print(
            "Classes:",
            sorted(int(class_id) for class_id in experience.classes_in_this_experience),
        )

        # Normal Avalanche training lifecycle.
        strategy.train(experience)

        print(f"========== Evaluation after experience {experience_index} ==========")

        # Normal Avalanche evaluation lifecycle.
        #
        # The standalone ML evaluator is trained automatically on all
        # accumulated frozen x,y evaluation memory before this evaluation.
        strategy.eval(benchmark.test_stream)

    # ------------------------------------------------------------------
    # Final results.
    # ------------------------------------------------------------------

    results = strategy.results()

    print()
    print("=== Summary ===")

    print(
        "mean_final_accuracy=",
        f"{results['mean_final_accuracy']:.4f}",
    )
    print(
        "mean_final_loss=",
        f"{results['mean_final_loss']:.4f}",
    )

    print("final_class_accuracy:")
    for class_id, accuracy in results["final_class_accuracy"].items():
        print(f"  class {class_id}: {accuracy:.4f}")

    print("final_class_loss:")
    for class_id, loss in results["final_class_loss"].items():
        print(f"  class {class_id}: {loss:.4f}")


if __name__ == "__main__":
    main()
