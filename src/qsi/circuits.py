"""재업로딩 회로와 큐비트별 관측량."""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from . import config


def build_circuit(n_qubits, layers=config.LAYERS, alpha=config.ALPHA):
    """특징별 RY 재업로딩과 사슬 얽힘을 구성한다."""
    qc = QuantumCircuit(n_qubits)
    x = ParameterVector("x", n_qubits)
    w = ParameterVector("w", 2 * n_qubits * layers)
    for layer in range(layers):
        for q in range(n_qubits):
            qc.ry(alpha * x[q], q)
        for q in range(n_qubits):
            qc.ry(w[2 * n_qubits * layer + 2 * q], q)
            qc.rz(w[2 * n_qubits * layer + 2 * q + 1], q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.barrier()
    return qc, x, w


def observables(n_qubits):
    """큐비트 0부터 순서대로 Pauli Z 관측량을 만든다."""
    return [SparsePauliOp("I" * (n_qubits - 1 - q) + "Z" + "I" * q) for q in range(n_qubits)]
