"""배치 상태벡터를 직접 계산하는 정확한 torch 시뮬레이터."""

import numpy as np
import torch


def apply_1q(state, U, q):
    """큐비트 한 축에만 2×2 게이트를 적용한다."""
    axis = state.ndim - 1 - q
    moved = state.movedim(axis, -1)
    if U.ndim == 2:
        result = torch.einsum("...j,ij->...i", moved, U)
    else:
        result = torch.einsum("b...j,bij->b...i", moved, U)
    return result.movedim(-1, axis)


def apply_cx(state, control, target):
    """제어 비트가 1인 진폭에만 표적 비트를 뒤집는다."""
    control_axis = state.ndim - 1 - control
    target_axis = state.ndim - 1 - target
    shape = [1] * state.ndim
    shape[control_axis] = 2
    mask = torch.tensor([False, True], device=state.device).reshape(shape)
    return torch.where(mask, state.flip(target_axis), state)


def _ry(theta):
    half = theta / 2
    c = torch.cos(half).to(torch.complex128)
    s = torch.sin(half).to(torch.complex128)
    return torch.stack((torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2)


def _rz(theta):
    half = theta / 2
    low = torch.exp(-1j * half)
    high = torch.exp(1j * half)
    zero = torch.zeros_like(low)
    return torch.stack((torch.stack((low, zero), dim=-1), torch.stack((zero, high), dim=-1)), dim=-2)


def statevector(x, w, n_qubits, layers, alpha):
    """Qiskit 리틀엔디언 순서의 복소 진폭을 반환한다."""
    x = x.to(torch.float64)
    w = w.to(torch.float64)
    batch = x.shape[0]
    state = torch.zeros((batch,) + (2,) * n_qubits, dtype=torch.complex128, device=x.device)
    state[(slice(None),) + (0,) * n_qubits] = 1
    for layer in range(layers):
        for q in range(n_qubits):
            state = apply_1q(state, _ry(alpha * x[:, q]), q)
        for q in range(n_qubits):
            state = apply_1q(state, _ry(w[2 * n_qubits * layer + 2 * q]), q)
            state = apply_1q(state, _rz(w[2 * n_qubits * layer + 2 * q + 1]), q)
        for q in range(n_qubits - 1):
            state = apply_cx(state, q, q + 1)
    return state.reshape(batch, 2 ** n_qubits)


def probabilities(x, w, n_qubits, layers, alpha):
    """각 계산 기저의 확률을 반환한다."""
    return statevector(x, w, n_qubits, layers, alpha).abs().square()


def expectations(x, w, n_qubits, layers, alpha):
    """큐비트별 정확한 Z 기댓값을 반환한다."""
    probs = probabilities(x, w, n_qubits, layers, alpha)
    basis = torch.arange(2 ** n_qubits, device=probs.device)
    signs = torch.stack([1 - 2 * ((basis >> q) & 1) for q in range(n_qubits)], dim=1)
    return probs @ signs.to(torch.float64)


def sample_expectations(x, w, n_qubits, layers, alpha, shots, seed):
    """다항 샘플로 측정한 Z 기댓값을 추정한다."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        probs = probabilities(x, w, n_qubits, layers, alpha).cpu().numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)
    counts = rng.multinomial(shots, probs)
    basis = np.arange(2 ** n_qubits)
    signs = np.stack([1 - 2 * ((basis >> q) & 1) for q in range(n_qubits)], axis=1)
    return counts @ signs / shots
