"""데이터 로드와 합금 기준 분할."""

import numpy as np
import pandas as pd

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """압축된 Si 영역 특징을 읽는다."""
    return pd.read_csv(path, compression="gzip")


def split_by_alloy(df):
    """평가 합금이 학습에 섞이지 않도록 분할한다."""
    test = df[df.split == "valid"].reset_index(drop=True)
    train_part = df[df.split == "train"]
    test_alloys = set(test.alloy)
    overlap = sorted(test_alloys & set(train_part.alloy))
    train = train_part[~train_part.alloy.isin(test_alloys)].reset_index(drop=True)
    return train, test, overlap


def threshold_set(train):
    """원 노트북과 같은 임계값 선택 표본을 뽑는다."""
    return train.sample(config.THRESHOLD_SET_SIZE, random_state=config.THRESHOLD_SET_SEED)


def balanced_sample(train, n_per_class, seed):
    """두 클래스를 동일 개수로 뽑고 원래 인덱스를 유지한다."""
    rng = np.random.default_rng(seed)
    u = train[train.usable == 1]
    idx = np.concatenate([rng.choice(u.index[u.label == c], n_per_class, replace=False) for c in (1, 0)])
    return train.loc[rng.permutation(idx)]
