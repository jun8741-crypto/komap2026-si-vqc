"""픽셀 및 영역 단위 평가 지표."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import config


def pixel_iou(df, pred):
    """영역별 정답 픽셀 수를 합산해 클래스별 IoU를 계산한다."""
    pred = np.asarray(pred).astype(bool)
    tp_p, tp_e = df.true_primary_px.to_numpy(), df.true_eutectic_px.to_numpy()
    inter_p = tp_p[pred].sum(); union_p = tp_p.sum() + tp_e[pred].sum()
    inter_e = tp_e[~pred].sum(); union_e = tp_e.sum() + tp_p[~pred].sum()
    iou_p = 100 * inter_p / union_p if union_p else 0.0
    iou_e = 100 * inter_e / union_e if union_e else 0.0
    return dict(iou_primary=iou_p, iou_eutectic=iou_e, miou=(iou_p + iou_e) / 2)


def blob_scores(df, pred):
    """사용 가능한 영역에 대해 F1·정확도·혼동행렬을 계산한다."""
    mask = df.usable.to_numpy() == 1
    y = df.label.to_numpy()[mask]
    p = np.asarray(pred).astype(int)[mask]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(f1=float(f1_score(y, p, zero_division=0)), accuracy=float(accuracy_score(y, p)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def evaluate(df, pred):
    """픽셀·영역 지표를 함께 반환한다."""
    return {**pixel_iou(df, pred), **blob_scores(df, pred)}


def pick_threshold(df, prob, grid=config.THRESHOLD_GRID):
    """최대 mIoU의 첫 임계값을 고른다."""
    values = [pixel_iou(df, np.asarray(prob) >= t)["miou"] for t in grid]
    return float(grid[np.argmax(values)])
