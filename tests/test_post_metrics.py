import numpy as np
import pandas as pd

from src.toolkit.post_metrics import compute_forgetting


def _frame(values):
    df = pd.DataFrame(
        {
            "seed": [0] * len(values),
            "mb_index": list(range(len(values))),
        }
    )
    # decorate_with_training_task expects the complete task-metric schema.
    for task in range(20):
        df[
            f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{task:03d}"
        ] = np.nan
    df["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"] = values
    return df


def test_compute_forgetting_uses_causal_maximum_not_first_value():
    df = _frame([0.20, 0.50, 0.40, 0.60, 0.30])

    result = compute_forgetting(
        df,
        "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000",
    )

    forgetting = result[
        "Forgetting_Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"
    ].tolist()

    assert np.isnan(forgetting[0])
    np.testing.assert_allclose(forgetting[1:], [0.0, 0.10, 0.0, 0.30])


def test_forgetting_does_not_use_future_accuracy():
    df = _frame([0.20, 0.50, 0.40])

    result = compute_forgetting(
        df,
        "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000",
    )

    forgetting = result[
        "Forgetting_Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"
    ].tolist()

    # The later 0.40 must not change the forgetting at the preceding 0.50.
    np.testing.assert_allclose(forgetting[1:], [0.0, 0.10])
