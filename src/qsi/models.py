"""로지스틱 기준선과 두 백엔드의 VQC 학습."""

from time import perf_counter

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN

from . import circuits, config, simulator


class VQCModel:
    """재업로딩 회로의 Z 기댓값에 선형 머리를 결합한다."""

    def __init__(self, n_qubits, features, layers=config.LAYERS, alpha=config.ALPHA, seed=0, backend="torch"):
        self.n_qubits = n_qubits
        self.features = list(features)
        self.layers = layers
        self.alpha = alpha
        self.seed = seed
        self.backend = backend
        self.scaler = StandardScaler()
        self.history = []
        self.train_seconds = 0.0
        init = np.random.default_rng(seed).uniform(-np.pi, np.pi, 2 * n_qubits * layers)
        torch.manual_seed(seed)
        if backend == "qiskit":
            qc, x, w = circuits.build_circuit(n_qubits, layers, alpha)
            est = StatevectorEstimator(default_precision=0.0, seed=seed)
            # 기본 precision은 기댓값에 샘플 잡음을 넣으므로 정확한 참값 비교에서는 0으로 고정한다.
            qnn = EstimatorQNN(circuit=qc, observables=circuits.observables(n_qubits),
                               input_params=list(x), weight_params=list(w), estimator=est,
                               gradient=ParamShiftEstimatorGradient(est), default_precision=0.0)
            self.net = torch.nn.Sequential(TorchConnector(qnn, initial_weights=init),
                                           torch.nn.Linear(n_qubits, 1))
        elif backend == "torch":
            self.weights = torch.nn.Parameter(torch.tensor(init, dtype=torch.float64))
            self.linear = torch.nn.Linear(n_qubits, 1).double()
        else:
            raise ValueError(f"지원하지 않는 백엔드: {backend}")

    def _input(self, df):
        values = np.clip(self.scaler.transform(df[self.features]), -config.CLIP, config.CLIP)
        dtype = torch.float64 if self.backend == "torch" else torch.float32
        return torch.tensor(values, dtype=dtype)

    def _forward(self, x):
        if self.backend == "torch":
            z = simulator.expectations(x, self.weights, self.n_qubits, self.layers, self.alpha)
            return self.linear(z).squeeze(-1)
        return self.net(x).squeeze(-1)

    def fit(self, sample, steps, batch=config.BATCH, lr=config.LR):
        """표준화 후 균형 표본에서 미니배치로 학습한다."""
        self.scaler.fit(sample[self.features])
        x = self._input(sample)
        y = torch.tensor(sample.label.to_numpy(), dtype=x.dtype)
        parameters = [self.weights, *self.linear.parameters()] if self.backend == "torch" else self.net.parameters()
        optimizer = torch.optim.Adam(parameters, lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        self.history = []
        start = perf_counter()
        for _ in range(steps):
            idx = rng.choice(len(x), batch, replace=False)
            optimizer.zero_grad()
            loss = criterion(self._forward(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach()))
        self.train_seconds = perf_counter() - start
        return self

    def _batches(self, df):
        x = self._input(df)
        for offset in range(0, len(x), config.PREDICT_BATCH):
            yield x[offset:offset + config.PREDICT_BATCH]

    def expectations(self, df):
        """입력 순서를 보존하며 회로 기댓값을 반환한다."""
        results = []
        with torch.no_grad():
            for x in self._batches(df):
                if self.backend == "torch":
                    result = simulator.expectations(x, self.weights, self.n_qubits, self.layers, self.alpha)
                else:
                    result = self.net[0](x)
                results.append(result.cpu().numpy())
        return np.concatenate(results) if results else np.empty((0, self.n_qubits))

    def predict_proba(self, df):
        """영역별 Primary 확률을 반환한다."""
        results = []
        with torch.no_grad():
            for x in self._batches(df):
                results.append(torch.sigmoid(self._forward(x)).cpu().numpy())
        return np.concatenate(results) if results else np.empty(0)

    def circuit_weights(self):
        """현재 회로 각도를 NumPy 배열로 반환한다."""
        weights = self.weights if self.backend == "torch" else self.net[0].weight
        return weights.detach().cpu().numpy().copy()

    def head(self):
        """선형 머리 계수와 절편을 반환한다."""
        linear = self.linear if self.backend == "torch" else self.net[1]
        return linear.weight.detach().cpu().numpy().reshape(-1).copy(), float(linear.bias.detach().item())

    def predict_proba_shots(self, df, shots, seed):
        """유한 샷으로 얻은 기댓값을 같은 머리로 판정한다."""
        weights = torch.tensor(self.circuit_weights(), dtype=torch.float64)
        coef, bias = self.head()
        results = []
        rng = np.random.default_rng(seed)
        for x in self._batches(df):
            shot_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            z = simulator.sample_expectations(x.double(), weights, self.n_qubits, self.layers,
                                              self.alpha, shots, shot_seed)
            results.append(expit(z @ coef + bias))
        return np.concatenate(results) if results else np.empty(0)


def fit_logistic(train_df, features):
    """표준화만 적용한 대회 기준선 확률 함수를 학습한다."""
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=2000)
    model.fit(scaler.fit_transform(train_df[features]), train_df.label)

    def predict(df):
        return model.predict_proba(scaler.transform(df[features]))[:, 1]

    return predict
