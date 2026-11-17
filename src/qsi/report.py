"""실험 결과를 표와 그림으로 저장한다."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, experiment, metrics


def _format(mean, std, digits=2):
    spread = "—" if pd.isna(std) else f"{std:.{digits}f}"
    return f"{mean:.{digits}f} ± {spread}"


def _overlap_note(rows):
    """문턱값 집합과 학습 데이터가 겹친 정도를 실제 실행 결과로 적는다."""
    note = (f"문턱값은 학습 분할에서 무작위로 뽑은 Si 영역 {config.THRESHOLD_SET_SIZE:,}개로 고른다. "
            "학습에 쓴 영역 중 이 집합과 겹친 수는 runs.csv의 thr_overlap에 있다")
    overlap = rows.groupby("key", sort=False).thr_overlap.max()
    parts = [f"{key} {int(value):,}개" for key, value in overlap.items() if value > 0]
    return note + (f"(조건별 최대: {', '.join(parts)})." if parts else ".")


def _write_markdown(rows, summary, verdicts, test, outdir):
    lines = ["# Si 영역 분류 비교", "", "| 조건 | 특징 수 | 학습 표본 | 걸음 | mIoU(%) | F1 | 정확도 | 학습 시간 평균(초) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.name} | {row.n_features} | {row.n_train} | {row.steps if row.steps else '—'} | "
                     f"{_format(row.miou_mean, row.miou_std)} | {_format(row.f1_mean, row.f1_std, 4)} | "
                     f"{_format(row.accuracy_mean, row.accuracy_std, 4)} | {row.train_seconds_mean:.2f} |")
    lines.extend(["", "## 비교 판정", ""])
    lines.extend(f"- {item['text']}" for item in verdicts)
    lines.extend(["", "같은 시드끼리 짝지은 mIoU 차이의 평균이 그 차이의 시드 표준편차의 2배를 넘고 모든 시드에서 방향이 같을 때만 차이로 본다."])
    baseline = metrics.evaluate(test, np.zeros(len(test), dtype=bool))
    lines.extend(["", f"기준선 ‘전부 Eutectic’: mIoU {baseline['miou']:.2f}%, 정확도 {baseline['accuracy']:.4f}.",
                  "", _overlap_note(rows),
                  "로지스틱은 대회 모델 그대로 표준화만 하고, VQC는 회전각 범위 때문에 표준화 뒤 ±3으로 자른다."])
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_loss(models, outdir):
    keys = list(dict.fromkeys(item[0] for item in models if isinstance(item, tuple)))
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        for item, artifact in models.items():
            if not isinstance(item, tuple):
                continue
            model_key, seed = item
            if model_key != key:
                continue
            history = np.asarray(artifact["history"])
            window = min(config.LOSS_WINDOW, len(history))
            if window:
                smoothed = np.convolve(history, np.ones(window) / window, mode="valid")
                ax.plot(np.arange(window - 1, len(history)), smoothed, label=f"seed {seed}")
        ax.set(title=key, xlabel="step", ylabel="loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_curves.png")
    plt.close(fig)


def _plot_miou(rows, outdir):
    fig, ax = plt.subplots(figsize=(max(7, rows.key.nunique() * 1.4), 4))
    keys = list(dict.fromkeys(rows.key))
    for i, key in enumerate(keys):
        values = rows.loc[rows.key == key, "miou"].to_numpy()
        ax.scatter(np.full(len(values), i), values, color="tab:blue")
        ax.hlines(values.mean(), i - 0.25, i + 0.25, color="tab:orange", linewidth=2)
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right")
    ax.set_ylabel("mIoU (%)")
    fig.tight_layout()
    fig.savefig(outdir / "miou_by_condition.png")
    plt.close(fig)


def _write_shots(models, outdir):
    candidates = [item for item in models if isinstance(item, tuple)]
    if not candidates:
        return
    selected = ("vqc7", 0) if ("vqc7", 0) in models else candidates[0]
    artifact = models[selected]
    model, threshold, test = artifact["model"], artifact["threshold"], artifact["test"]
    sample = test.sample(min(config.SHOT_SAMPLE_SIZE, len(test)), random_state=config.SHOT_SAMPLE_SEED)
    exact = model.predict_proba(sample)
    exact_pred = exact >= threshold
    exact_miou = metrics.pixel_iou(sample, exact_pred)["miou"]
    rows = []
    for shots in config.SHOTS:
        sampled = model.predict_proba_shots(sample, shots, config.SHOT_SAMPLE_SEED)
        shot_pred = sampled >= threshold
        rows.append(dict(model=selected[0], shots=shots, mean_abs_err=float(np.mean(np.abs(sampled - exact))),
                         flip_percent=float(100 * np.mean(shot_pred != exact_pred)),
                         miou_shots=metrics.pixel_iou(sample, shot_pred)["miou"], miou_exact=exact_miou))
    pd.DataFrame(rows).to_csv(outdir / "shots.csv", index=False)


def write_results(rows, verdicts, models, outdir, figures=True):
    """행·요약·판정·그림·샷 민감도를 파일로 기록한다."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(rows)
    summary = experiment.summarize(rows)
    rows.to_csv(outdir / "runs.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    _write_markdown(rows, summary, verdicts, models["_test"], outdir)
    if figures:
        _plot_miou(rows, outdir)
        _plot_loss(models, outdir)
        _write_shots(models, outdir)
