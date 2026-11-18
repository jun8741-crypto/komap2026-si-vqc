"""조건별 실행, 집계 및 시드 간 비교 판정."""

from time import perf_counter

import numpy as np
import pandas as pd

from . import config, io, metrics, models


class RunResults(list):
    """결과 행과 그림·샷 분석용 모델을 함께 보관한다."""

    def __init__(self):
        super().__init__()
        self.models = {}


def run_condition(cond, train, test, thr_set, seeds=config.SEEDS, steps=None, backend="torch") -> list[dict]:
    """조건을 지정한 시드에서 실행하고 평가 행을 반환한다."""
    results = RunResults()
    current_seeds = seeds if cond["seeded"] else (-1,)
    for seed in current_seeds:
        sample = (train[train.usable == 1] if cond["training"] == "all"
                  else io.balanced_sample(train, cond["n_per_class"], seed))
        start = perf_counter()
        if cond["kind"] == "logistic":
            predict = models.fit_logistic(sample, cond["features"])
            model = None
            history = []
            actual_steps = 0
            train_seconds = perf_counter() - start
        else:
            actual_steps = cond["steps"] if steps is None else steps
            model = models.VQCModel(len(cond["features"]), cond["features"], seed=seed, backend=backend)
            model.fit(sample, actual_steps)
            predict = model.predict_proba
            history = model.history
            train_seconds = model.train_seconds
        threshold = metrics.pick_threshold(thr_set, predict(thr_set))
        test_prob = predict(test)
        scores = metrics.evaluate(test, test_prob >= threshold)
        row = dict(key=cond["key"], name=cond["name"], kind=cond["kind"],
                   n_features=len(cond["features"]), seed=seed, n_train=len(sample),
                   steps=actual_steps, train_seconds=train_seconds, threshold=threshold,
                   **scores, final_loss=float(np.mean(history[-config.FINAL_LOSS_WINDOW:])) if history else np.nan,
                   thr_overlap=len(sample.index.intersection(thr_set.index)))
        results.append(row)
        if model is not None:
            results.models[(cond["key"], seed)] = dict(model=model, threshold=threshold,
                                                       history=history, test=test, test_prob=test_prob)
    return results


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """조건별 평균과 표본표준편차를 계산한다."""
    columns = ["miou", "f1", "accuracy", "train_seconds"]
    aggregate = {column: ["mean", "std"] for column in columns}
    summary = rows.groupby("key", sort=False).agg(aggregate)
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index()
    present = [column for column in ("name", "n_features", "n_train", "steps") if column in rows]
    metadata = rows.groupby("key", sort=False)[present].first().reset_index()
    metadata["n_seeds"] = rows.groupby("key", sort=False).size().to_numpy()
    return metadata.merge(summary, on="key", sort=False)


def verdict(rows, a_key, b_key) -> dict:
    """같은 시드의 mIoU 차이를 기준으로 보수적으로 판정한다."""
    a = rows[rows.key == a_key].set_index("seed").miou
    b = rows[rows.key == b_key].set_index("seed").miou
    if -1 in a.index and -1 in b.index:
        diffs = np.array([a.loc[-1] - b.loc[-1]], dtype=float)
    elif -1 in a.index:
        diffs = a.loc[-1] - b.to_numpy(dtype=float)
    elif -1 in b.index:
        diffs = a.to_numpy(dtype=float) - b.loc[-1]
    else:
        common = a.index.intersection(b.index)
        diffs = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    mean = float(np.mean(diffs)) if len(diffs) else float("nan")
    noise = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    significant = bool(len(diffs) > 1 and abs(mean) > config.VERDICT_MULTIPLIER * noise
                       and np.all(diffs * mean > 0))
    if len(diffs) < 2:
        ending = "시드가 2개 미만이라 판정하지 않음"
    elif significant:
        ending = f"{a_key if mean > 0 else b_key}가 높음"
    else:
        ending = "시드 간 변동과 구별되지 않음"
    noise_text = "—" if np.isnan(noise) else f"{noise:.2f}"
    text = f"{a_key} − {b_key}: 평균 {mean:+.2f} (차이의 시드 표준편차 {noise_text}) → {ending}"
    return dict(a=a_key, b=b_key, diffs=diffs.tolist(), mean=mean, noise=noise,
                significant=significant, text=text)
