import numpy as np
import pytest

from qsi import config, io, models


@pytest.mark.parametrize("features", [config.FEATURES2, config.FEATURES7])
def test_torch_training_improves_loss(data_splits, features):
    train, _, _ = data_splits
    model = models.VQCModel(len(features), features, seed=0)
    model.fit(io.balanced_sample(train, 32, 0), 20)
    assert len(model.history) == 20
    assert np.isfinite(model.history).all()
    assert np.mean(model.history[-5:]) < np.mean(model.history[:5])
    assert np.all((model.predict_proba(train.head(3)) >= 0) & (model.predict_proba(train.head(3)) <= 1))


def test_backend_initialization_and_training_match(data_splits):
    train, _, _ = data_splits
    a = models.VQCModel(2, config.FEATURES2, seed=0, backend="torch")
    b = models.VQCModel(2, config.FEATURES2, seed=0, backend="qiskit")
    assert np.max(np.abs(a.head()[0] - b.head()[0])) < 1e-7
    assert abs(a.head()[1] - b.head()[1]) < 1e-7
    assert np.max(np.abs(a.circuit_weights() - b.circuit_weights())) < 1e-6
    sample = io.balanced_sample(train, 256, 0)
    a.fit(sample, 10)
    b.fit(sample, 10)
    assert np.max(np.abs(np.array(a.history) - b.history)) < 1e-4
    assert np.max(np.abs(a.circuit_weights() - b.circuit_weights())) < 1e-4


def test_logistic_reference_probabilities(data_splits):
    train, test, _ = data_splits
    usable = train[train.usable == 1]
    for features, expected in [(config.FEATURES2, [0.01897317, 0.02100065, 0.01846158]),
                               (config.FEATURES7, [0.03172296, 0.02358067, 0.01254024])]:
        prob = models.fit_logistic(usable, features)(test.head(3))
        assert np.max(np.abs(prob - expected)) < 1e-6


def test_batched_prediction_and_shots_converge(data_splits):
    # 예측 배치(PREDICT_BATCH)를 넘는 입력도 한 행씩 계산한 값과 같고, 샷을 늘리면 정확한 확률에 가까워진다.
    train, test, _ = data_splits
    model = models.VQCModel(7, config.FEATURES7, seed=0).fit(io.balanced_sample(train, 32, 0), 5)
    big = test.head(config.PREDICT_BATCH + 10)
    full = model.predict_proba(big)
    tail = model.predict_proba(big.tail(10))
    assert np.max(np.abs(full[-10:] - tail)) < 1e-12
    exact = model.predict_proba(test.head(50))
    shots = model.predict_proba_shots(test.head(50), 200000, seed=0)
    assert np.mean(np.abs(shots - exact)) < 0.01
