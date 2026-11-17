import numpy as np
import pytest
import torch
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

from qsi import circuits, simulator


@pytest.mark.parametrize("n", [2, 7])
def test_statevector_expectations_and_gradient_match_reference(n):
    rng = np.random.default_rng(42)
    x_np = rng.uniform(-3, 3, (5, n))
    w_np = rng.uniform(-np.pi, np.pi, 4 * n)
    x = torch.tensor(x_np, dtype=torch.float64)
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    qc, xp, wp = circuits.build_circuit(n)
    obs = circuits.observables(n)
    refs = [Statevector(qc.assign_parameters({**dict(zip(xp, xi)), **dict(zip(wp, w_np))})) for xi in x_np]
    state = simulator.statevector(x, w, n, 2, 1.0).detach().numpy()
    assert np.max(np.abs(state - np.stack([s.data for s in refs]))) < 1e-10
    got = simulator.expectations(x, w, n, 2, 1.0)
    expected = np.array([[s.expectation_value(o).real for o in obs] for s in refs])
    assert np.max(np.abs(got.detach().numpy() - expected)) < 1e-10
    got.sum().backward()
    h = 1e-6
    fd = np.empty_like(w_np)
    for i in range(len(fd)):
        plus, minus = w_np.copy(), w_np.copy()
        plus[i] += h
        minus[i] -= h
        fd[i] = (simulator.expectations(x, torch.tensor(plus), n, 2, 1.0).sum().item()
                 - simulator.expectations(x, torch.tensor(minus), n, 2, 1.0).sum().item()) / (2 * h)
    assert np.max(np.abs(w.grad.numpy() - fd)) < 1e-6


def test_qnn_forward_backward_match_torch():
    n = 2
    rng = np.random.default_rng(3)
    x_np, w_np = rng.uniform(-3, 3, (2, n)), rng.uniform(-np.pi, np.pi, 4 * n)
    qc, xp, wp = circuits.build_circuit(n)
    est = StatevectorEstimator(default_precision=0.0, seed=0)
    qnn = EstimatorQNN(circuit=qc, observables=circuits.observables(n), input_params=list(xp),
                       weight_params=list(wp), estimator=est,
                       gradient=ParamShiftEstimatorGradient(est), default_precision=0.0)
    x = torch.tensor(x_np, dtype=torch.float64)
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    result = simulator.expectations(x, w, n, 2, 1.0)
    result.sum().backward()
    assert np.max(np.abs(qnn.forward(x_np, w_np) - result.detach().numpy())) < 1e-8
    assert np.max(np.abs(qnn.backward(x_np, w_np)[1].sum(axis=(0, 1)) - w.grad.numpy())) < 1e-8


def test_probabilities_and_shots():
    x = torch.tensor([[0.2, -0.8]], dtype=torch.float64)
    w = torch.zeros(8, dtype=torch.float64)
    assert abs(simulator.probabilities(x, w, 2, 2, 1).sum().item() - 1) < 1e-12
    exact = simulator.expectations(x, w, 2, 2, 1).numpy()
    sampled = simulator.sample_expectations(x, w, 2, 2, 1, 200000, 0)
    assert np.max(np.abs(sampled - exact)) < 0.02


@pytest.mark.parametrize("n, layers, alpha", [(3, 3, 0.7), (4, 1, 1.3)])
def test_statevector_matches_reference_for_other_depths_and_scales(n, layers, alpha):
    # layers·alpha가 기본값이 아닐 때도 각도 색인과 입력 배율이 qiskit 회로와 같은지 확인한다.
    rng = np.random.default_rng(7)
    x_np = rng.uniform(-3, 3, (4, n))
    w_np = rng.uniform(-np.pi, np.pi, 2 * n * layers)
    qc, xp, wp = circuits.build_circuit(n, layers, alpha)
    refs = np.stack([Statevector(qc.assign_parameters({**dict(zip(xp, xi)), **dict(zip(wp, w_np))})).data
                     for xi in x_np])
    got = simulator.statevector(torch.tensor(x_np), torch.tensor(w_np), n, layers, alpha).numpy()
    assert np.max(np.abs(got - refs)) < 1e-10
