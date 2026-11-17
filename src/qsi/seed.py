"""난수 재현성 관리."""

import random
import numpy as np
import torch


def set_seed(seed):
    """파이썬, NumPy, torch의 난수 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
