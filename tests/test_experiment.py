import numpy as np
import pandas as pd

from qsi import experiment
from qsi import config


def test_verdict_pairing_and_significance():
    rows = pd.DataFrame([{"key": key, "seed": seed, "miou": value}
                         for key, values in {"a": [5, 6, 7], "b": [1, 2, 3],
                                             "c": [4, 6, 8], "d": [6, 5, 6]}.items()
                         for seed, value in enumerate(values)])
    assert experiment.verdict(rows, "a", "b")["significant"]
    assert not experiment.verdict(rows, "a", "c")["significant"]
    assert not experiment.verdict(rows, "a", "d")["significant"]
    one = rows[rows.seed == 0]
    result = experiment.verdict(one, "a", "b")
    assert not result["significant"] and np.isnan(result["noise"])
    assert "시드가 2개 미만" in result["text"]


def test_summary_uses_sample_standard_deviation():
    rows = pd.DataFrame({"key": ["a", "a"], "miou": [1., 3.], "f1": [0., 1.],
                         "accuracy": [0., 1.], "train_seconds": [2., 4.]})
    result = experiment.summarize(rows).iloc[0]
    assert result.miou_mean == 2
    assert abs(result.miou_std - np.sqrt(2)) < 1e-12


def test_condition_returns_test_probabilities_for_reporting(data_splits):
    train, test, _ = data_splits
    cond = dict(key="vqc2", name="VQC 2큐비트", kind="vqc", features=config.FEATURES2,
                training="sample", seeded=True, n_per_class=32, steps=1)
    result = experiment.run_condition(cond, train, test.head(5), train.head(10), seeds=(0,))
    assert result.models[("vqc2", 0)]["test_prob"].shape == (5,)


def test_verdict_against_unseeded_condition_and_lower_side():
    # 시드 없는 조건(seed=-1)은 모든 시드와 짝지어지고, b가 높으면 b를 이긴 쪽으로 적는다.
    rows = pd.DataFrame([{"key": "vqc", "seed": s, "miou": v} for s, v in enumerate([88.0, 88.2, 87.9])]
                        + [{"key": "logi", "seed": -1, "miou": 90.0}])
    result = experiment.verdict(rows, "vqc", "logi")
    assert np.allclose(result["diffs"], [-2.0, -1.8, -2.1])
    assert result["significant"] and result["text"].endswith("logi가 높음")
    reverse = experiment.verdict(rows, "logi", "vqc")
    assert np.allclose(reverse["diffs"], [2.0, 1.8, 2.1]) and reverse["text"].endswith("logi가 높음")
