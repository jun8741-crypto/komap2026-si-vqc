import numpy as np
import pandas as pd

from qsi import io, metrics


def test_pixel_and_blob_metrics_on_three_rows():
    df = pd.DataFrame({"true_primary_px": [10, 0, 2], "true_eutectic_px": [0, 10, 8],
                       "usable": [1, 1, 0], "label": [1, 0, -1]})
    result = metrics.evaluate(df, [1, 1, 0])
    assert np.isclose(result["iou_primary"], 100 * 10 / 22)
    assert np.isclose(result["iou_eutectic"], 100 * 8 / 20)
    assert result["f1"] == 2 / 3
    assert result["accuracy"] == 0.5
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (0, 1, 0, 1)


def test_all_eutectic_baseline(data_splits):
    _, test, _ = data_splits
    result = metrics.evaluate(test, np.zeros(len(test)))
    assert abs(result["miou"] - 19.3098) < 0.0001
    assert abs(result["accuracy"] - 0.94272) < 0.00001


def test_threshold_uses_first_maximum():
    df = pd.DataFrame({"true_primary_px": [10, 0], "true_eutectic_px": [0, 10]})
    assert abs(metrics.pick_threshold(df, [0.9, 0.1]) - 0.11) < 1e-9
