"""실험 경로와 조건의 단일 정의."""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/si_blob_features.csv.gz"
RESULTS_DIR = ROOT / "results"
FEATURES2 = ["p_primary", "p_si3"]
FEATURES7 = ["p_primary", "p_si3", "norm_thickness", "log_area", "elongation", "thickness_std", "perimeter_ratio"]
SEEDS = (0, 1, 2)
THRESHOLD_SET_SIZE = 5000
THRESHOLD_SET_SEED = 0
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
N_PER_CLASS = 256
STEPS = 100
BATCH = 64
LR = 0.05
LAYERS = 2
ALPHA = 1.0
CLIP = 3.0
LONG_N_PER_CLASS = 1024
LONG_STEPS = 500
PREDICT_BATCH = 4096
SHOT_SAMPLE_SIZE = 2000
SHOT_SAMPLE_SEED = 1
SHOTS = (32, 128, 512, 2048, 8192)
LOSS_WINDOW = 5
FINAL_LOSS_WINDOW = 10
VERDICT_MULTIPLIER = 2

CONDITIONS = [
    dict(key="logi2_all", name="로지스틱 2특징(전체)", kind="logistic", features=FEATURES2, training="all", seeded=False, n_per_class=None, steps=0),
    dict(key="logi7_all", name="로지스틱 7특징(전체, 대회 모델)", kind="logistic", features=FEATURES7, training="all", seeded=False, n_per_class=None, steps=0),
    dict(key="logi2_s512", name="로지스틱 2특징(표본 512)", kind="logistic", features=FEATURES2, training="sample", seeded=True, n_per_class=N_PER_CLASS, steps=0),
    dict(key="logi7_s512", name="로지스틱 7특징(표본 512)", kind="logistic", features=FEATURES7, training="sample", seeded=True, n_per_class=N_PER_CLASS, steps=0),
    dict(key="logi7_s2048", name="로지스틱 7특징(표본 2048)", kind="logistic", features=FEATURES7, training="sample", seeded=True, n_per_class=LONG_N_PER_CLASS, steps=0),
    dict(key="vqc2", name="VQC 2큐비트", kind="vqc", features=FEATURES2, training="sample", seeded=True, n_per_class=N_PER_CLASS, steps=STEPS),
    dict(key="vqc7", name="VQC 7큐비트", kind="vqc", features=FEATURES7, training="sample", seeded=True, n_per_class=N_PER_CLASS, steps=STEPS),
    dict(key="vqc7_long", name="VQC 7큐비트(긴 학습)", kind="vqc", features=FEATURES7, training="sample", seeded=True, n_per_class=LONG_N_PER_CLASS, steps=LONG_STEPS),
]
VERDICT_PAIRS = [("vqc2", "logi2_s512"), ("vqc2", "logi2_all"), ("vqc7", "logi7_s512"),
                 ("vqc7", "logi7_all"), ("vqc7", "vqc2"), ("vqc7_long", "logi7_s2048"),
                 ("vqc7_long", "vqc7"), ("vqc7_long", "logi7_all")]
