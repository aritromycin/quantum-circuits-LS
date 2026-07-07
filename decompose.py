"""Decompose an arbitrary unitary into the universal basis {H, T, CNOT}.

A unitary is
lowered in stages:

    1. twolevel_decomposition:  Unitary     -> list[TwoLevel]
    2. decompose_twolevel:      TwoLevel    -> SingleQubitGate + ControlledU
    3. decompose_controlledU:   ControlledU -> CU + CNOT
    4. decompose_cu:            CU          -> SingleQubitGate + CNOT
    5. decompose_to_ht:         SingleQubitGate -> H / T words (using rotation.py)

Numpy types: a `Unitary` (N x N) and a 2x2 gate block are
both np.ndarray (complex128); a `ComplexVec` is a 1-D np.ndarray. A `Circuit` is a
Python list of gate objects, each exposing `to_unitary()`; gates are stored in
order of application (the first gate is applied first, i.e. the rightmost matrix
factor).

Every function/method below is a stub for you to implement; See "03 - Completing the Decomposition.pdf" for the recommended order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

import rotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def num_qubits(N: int) -> int:
    """Number of qubits n such that N == 2^n (N is the unitary / two-level size)."""
    # TODO: implement.
    n = int(np.log2(N))
    return n


# ---------------------------------------------------------------------------
# Gate representations
#
# Each is a sparse description of an operation with a `to_unitary()` returning the
# full N x N matrix. As the decomposition progresses, gates get rewritten into
# simpler ones. The 2x2 block `unitary` is a (2, 2) complex ndarray.
# ---------------------------------------------------------------------------


@dataclass
class TwoLevel:
    """A two-level unitary: acts as the 2x2 `unitary` on the two basis states
    `level0`, `level1` of a size-`size` register, and as identity everywhere else.
    """

    size: int
    level0: int
    level1: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand to the full `size` x `size` matrix: identity except the 2x2 block
        placed at rows/cols (level0, level1).
        """
        # TODO: implement.
        mat = np.eye(size, dtype=complex)
        mat[self.level0, self.level0] = self.unitary[0,0]
        mat[self.level0, self.level1] = self.unitary[0,1]
        mat[self.level1, self.level0] = self.unitary[1,0]
        mat[self.level1, self.level1] = self.unitary[1,1]
        return mat


@dataclass
class SingleQubitGate:
    """A single-qubit gate acting as the 2x2 `unitary` on `qubit` of an n-qubit
    register (N = 2^n), identity on the other qubits.
    """

    n: int
    qubit: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Expand the 2x2 to N dimensions: for each basis index whose `qubit` bit is
        0, fill the 2x2 block linking it to its partner (that bit = 1).
        """
        # TODO: implement.
        N = 2**self.n
        mat = np.eye(N, dtype=complex)
        for i in range(N):
            bit = (i >> self.qubit) & 1
            if (bit == 0):
                mask = 1 << self.qubit
                j = i ^ mask
                mat[i,i] = self.unitary[0,0]
                mat[i,j] = self.unitary[0,1]
                mat[j,i] = self.unitary[1,0]
                mat[j,j] = self.unitary[1,1]
        return mat


@dataclass
class ControlledU:
    """A fully-controlled single-qubit gate C^k(U): apply the 2x2 `unitary` to
    `target` iff every other qubit is 1. Controls are always conditioned on 1, so
    their positions need not be stored.
    """

    n: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity everywhere except the single controlled block: the pair (all
        ones except the target bit, all ones).
        """
        # TODO: implement.
        N = 2**self.n
        mat = np.eye(N,dtype=complex)
        j = N-1
        i = j-(2**(self.target))
        mat[i,i] = self.unitary[0,0]
        mat[i,j] = self.unitary[0,1]
        mat[j,i] = self.unitary[1,0]
        mat[j,j] = self.unitary[1,1]
        return mat


@dataclass
class CU:
    """A singly-controlled single-qubit gate C(U): apply the 2x2 `unitary` to
    `target` iff `control` is 1. The full U(2) (global phase kept) is stored, since
    under a control the global phase becomes a physical relative phase. This is the
    recursion leaf of decompose_controlled.
    """

    n: int
    control: int
    target: int
    unitary: np.ndarray  # (2, 2)

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where `unitary` acts on `target`."""
        # TODO: implement.
        N = 2**self.n
        mat = np.eye(N, dtype=complex)
        for i in range(N):
            tbit = (i >> self.target) & 1
            cbit = (i >> self.control) & 1
            if (tbit == 0 and cbit == 1):
                mask = 1 << self.target
                j = i ^ mask
                mat[i,i] = self.unitary[0,0]
                mat[i,j] = self.unitary[0,1]
                mat[j,i] = self.unitary[1,0]
                mat[j,j] = self.unitary[1,1]
        return mat


@dataclass
class CNOT:
    """A controlled-NOT: flip `target` iff `control` is 1. Its 2x2 is fixed to
    Pauli-X, so (unlike CU) it stores no unitary.
    """

    n: int
    control: int
    target: int

    def to_unitary(self) -> np.ndarray:
        """Identity except the control=1 blocks, where X swaps the target's 0/1
        amplitudes.
        """
        # TODO: implement.
        N = 2**self.n
        mat = np.eye(N, dtype=complex)
        for i in range(N):
            tbit = (i >> self.target) & 1
            cbit = (i >> self.control) & 1
            if (tbit == 0 and cbit == 1):
                mask = 1 << self.target
                j = i ^ mask
                mat[i,i] = 0
                mat[i,j] = 1
                mat[j,i] = 1
                mat[j,j] = 0
        return mat
            
        raise NotImplementedError("CNOT.to_unitary is not implemented yet")


@dataclass
class Swap:
    """A multi-controlled NOT (generalized Toffoli): flip `target` iff every other
    qubit equals its entry in `control_vals`. `control_vals` has size n and is
    indexed by qubit; control_vals[target] is unused.
    """

    target: int
    control_vals: list[bool]

    def to_unitary(self) -> np.ndarray:
        N = 2**len(control_vals)
        mat = np.eye(N, dtype=complex)
        for i in range(N):
            tbit = (i >> self.target) & 1
            if (all(((i >> q) & 1) == int(self.control_vals[q]) for q in range(len(control_vals)) if q != self.target) and tbit == 0):
                mask = 1 << self.target
                j = i ^ mask
                mat[i,i] = 0
                mat[i,j] = 1
                mat[j,i] = 1
                mat[j,j] = 0
        return mat
                


# A gate is any of the sparse representations above; a circuit is a list of gates.
Gate = Union[TwoLevel, SingleQubitGate, ControlledU, CU, CNOT]
Circuit = list  # list[Gate]
TwoLevels = list  # list[TwoLevel]


def circuit_to_unitary(circuit: Circuit) -> np.ndarray:
    """Full N x N unitary of a whole circuit. Gates are stored in order of
    application, so the product premultiplies (first gate is the rightmost factor):
    result = g_last @ ... @ g_1. Assumes the circuit is non-empty.
    """
    # TODO: implement.
    g_total = np.eye(N, dtype=complex)
    for i in Circuit:
        mygate = i.to_unitary()
        g_total = mygate @ g_total
    return g_total


def to_circuit(two_levels: TwoLevels) -> Circuit:
    """Wrap a two-level sequence as a circuit, so decompose_unitary /
    twolevel_decomposition output flows straight into a Circuit.
    """
    # TODO: implement.
    mycircuit = Circuit(two_levels)
    return mycircuit


def error_up_to_phase(a: np.ndarray, b: np.ndarray) -> float:
    """Elementwise difference between two same-size unitaries, ignoring an overall
    global phase: align b to a by the phase of their Hilbert-Schmidt overlap
    <b, a> = sum conj(b_ij) a_ij, then compare. ~0 means equal up to global phase.
    """
    # TODO: implement.
    overlap = np.sum(np.conj(b),a)
    phase = overlap/np.abs(overlap)
    b_aligned = phase*b
    compare = np.linalg.norm(b_aligned - a)
    return compare


# ---------------------------------------------------------------------------
# Stage 1: Unitary -> two-level unitaries (see cpp/src/TwoLevel.h)
# ---------------------------------------------------------------------------


def align(x: complex, y: complex, norm: float) -> np.ndarray:
    """The 2x2 unitary [[conj(x), conj(y)], [-y, x]] / norm. Premultiplying it onto
    a column with entries (x, y) at two levels rotates the amplitude at the second
    level onto the first, leaving the real `norm` there and 0 below.
    """
    # TODO: implement.
    my_mat = np.array([[np.conj(x), np.conj(y)], [-y, x]], dtype=complex)
    return my_mat/norm


def decompose_vector(vec: np.ndarray) -> TwoLevels:
    """Given the first column of a unitary, return a sequence of two-levels which,
    when premultiplied onto the unitary, make its first column be (1, 0, 0, ...).
    Walk from the bottom up, using `align` at each pivot to zero out one entry; the
    running pivot holds the accumulated real norm after the first rotation.
    """
    # TODO: implement.
    n = len(vec)
    v = vec.astype(complex).copy()
    L = []
    for i in range(n-1,0,-1):
        norm = np.sqrt(np.abs(v[i]**2) + np.abs(v[i-1]**2))
        if norm == 0:
            continue
        mat = align(v[i-1],v[i],norm)
        L.append(TwoLevel(size=n, level0=i-1, level1=i, unitary=mat))
        v[i-1] = norm
        v[i] = 0
    return L


def expand_twolevels(input: TwoLevels, n: int) -> TwoLevels:
    """Expand each TwoLevel to n dimensions by shifting size, level0, level1 up by
    the offset (n - tl.size). Used to lift a sub-block decomposition back to full n.
    """
    # TODO: implement.
    output = []
    for i in input:
        output.append(TwoLevel(size=n, level0=i.level0+n-i.size, level1=i.level1+n-i.size, unitary=i.unitary))
    return output


def two_levels_to_unitary(two_levels: TwoLevels) -> np.ndarray:
    """Full matrix of a two-level sequence: premultiply each two-level's matrix in
    order (result = tl.to_unitary() @ result), reproducing the application order.
    """
    # TODO: implement.
    my_mat = np.eye(two_levels[0].size,dtype=complex)
    for i in two_levels:
        my_mat = i.to_unitary() @ my_mat
    return my_mat


def adjoint_twolevel(tl: TwoLevel) -> TwoLevel:
    """Adjoint of a single two-level: same levels, adjoint (conjugate transpose) of
    the 2x2 block.
    """
    # TODO: implement.
    L = TwoLevel(size=tl.size,level0=tl.level0,level1=tl.level1,unitary=tl.unitary.conj().T)
    return L


def adjoint_twolevels(two_levels: TwoLevels) -> TwoLevels:
    """Adjoint of a sequence: reverse the order and take the adjoint of each, since
    (A_k ... A_1)^dagger = A_1^dagger ... A_k^dagger.
    """
    # TODO: implement.
    k = len(two_levels)
    L = []
    for i in range(k-1,-1,-1):
       L.append(adjoint_twolevel(two_levels[i])) 
    return L


def decompose_unitary(u: np.ndarray) -> TwoLevels:
    """Repeat decompose_vector on successive sub-columns to reduce u to identity.
    At step k, columns/rows 0..k-1 are already reduced, so work on the lower-right
    (n-k) block: clear column k below the diagonal. Finally append a phase two-level
    on the last two levels to cancel the residual phase, so the product is identity.
    Returns the sequence S with prod(S) @ u == I (i.e. prod(S) = u^dagger).
    """
    # TODO: implement.
    L = []
    n = len(u)
    u_copy = u.astype(complex).copy()
    for i in range(n):
        vec = u_copy[i:,i]
        tls = decompose_vector(vec)
        L.extend(expand_twolevels(tls,n))
        for mat in tls:
            u_copy = mat.to_unitary() @ u_copy
    phase = u[n-1,n-1]
    phase_fix = np.array([[1,0],[0,np.conj(phase)]], dtype=complex)
    L.append(TwoLevel(size=n,level0=n-2,level1=n-1,unitary=phase_fix))
    return L


def twolevel_decomposition(u: np.ndarray) -> TwoLevels:
    """The two-level decomposition of u itself: decompose_unitary returns the
    sequence S that reduces u to identity (prod(S) = u^dagger), so its adjoint is
    the sequence whose product is u.
    """
    return adjoint_twolevels(decompose_unitary(u))


# ---------------------------------------------------------------------------
# ABC decomposition of a single-qubit gate (see cpp/src/ABC.h)
# ---------------------------------------------------------------------------


@dataclass
class ABC:
    """Nielsen & Chuang Corollary 4.2: every single-qubit U factors as
    U = e^{i alpha} A X B X C with A B C = I (X is Pauli-X). Building block for a
    single-controlled C(U).
    """

    alpha: float  # global phase
    A: np.ndarray  # (2, 2)
    B: np.ndarray  # (2, 2)
    C: np.ndarray  # (2, 2)


def abc_decompose(u: np.ndarray) -> ABC:
    """Build the ABC decomposition of u (Corollary 4.2). Take the ZYZ Euler angles
    (alpha, beta, gamma, delta) of u, then set
        A = Rz(beta) Ry(gamma/2)
        B = Ry(-gamma/2) Rz(-(delta+beta)/2)
        C = Rz((delta-beta)/2)
    Using X Ry(t) X = Ry(-t) and X Rz(t) X = Rz(-t), these satisfy A B C = I and
    e^{i alpha} A X B X C = u.
    """
    # TODO: implement using rotation.euler_angles_zyz, rotation.Rz/Ry.
    angles = rotation.zyz_angles(rotation.to_bloch(u))
    alpha = angles[3]
    beta = angles[0]
    gamma = angles[1]
    delta = angles[2]
    A = rotation.R_form(beta,np.array([0,0,1])) @ rotation.R_form(gamma/2,np.array([0,1,0]))
    B = rotation.R_form(-gamma/2,np.array([0,1,0])) @ rotation.R_form(-(delta+beta)/2,np.array([0,0,1]))
    C = rotation.R_form((delta-beta)/2, np.array([0,0,1]))
    myobj = ABC(alpha=alpha,A=A,B=B,C=C)
    return myobj


def abc_reconstruct(d: ABC) -> np.ndarray:
    """Reassemble e^{i alpha} A X B X C from an ABC (inverse of abc_decompose)."""
    # TODO: implement.
    X = np.array([[0,1],[1,0]],dtype=complex)
    mat = np.exp(1j*d.alpha)*(d.A @ X @ d.B @ X @ d.C)
    return mat


# ---------------------------------------------------------------------------
# Gray code and controlled circuits (see cpp/src/Swap.h, cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def gray_code(tl: TwoLevel) -> list[Swap]:
    """Gray code connecting level0 and level1 of a two-level, as the sequence of
    single-qubit flips walking from level0 to level1 (one Swap per differing bit).
    At each step the Swap records which qubit flips and the current code's values on
    the other qubits (the control pattern).
    """
    # TODO: implement.
    n = num_qubits(tl.size)
    current = tl.level0
    diff = current ^ tl.level1
    diff_bits = [q for q in range(n) if (diff >> q) & 1]
    L = []
    for i in diff_bits:
        control_values = [bool((current >> q) & 1) for q in range(n)]
        L.append(Swap(target=i,control_vals=control_values))
    return L


def decompose_swap(swap: Swap) -> Circuit:
    """Decompose a Swap (multi-controlled NOT) into a Circuit: a controlled-X with
    the swap's arbitrary control values.
    """
    # TODO: implement (hint: controlled_circuit with Pauli-X).
    n = len(swap.control_vals)
    X = np.array([[0,1],[1,0]],dtype=complex)
    return controlled_circuit(n,swap.target,swap.control_vals,X)


def controlled_circuit(n: int, target: int, control_vals: list[bool], unitary: np.ndarray) -> Circuit:
    """Circuit applying the 2x2 `unitary` to `target` iff every non-target qubit
    equals control_vals[q]. Realized as a fully-controlled-U core (ControlledU,
    controls = all 1) sandwiched by X gates on the qubits conditioned on 0, so those
    become 1-controls. The sandwich is symmetric (X is its own inverse).
    """
    # TODO: implement.
    X = np.array([[0,1],[1,0]],dtype=complex)
    zero_condition_qubits = [q for q in range(n) if q!=target and not control_vals[q]]
    X_gates = [SingleQubitGate(n=n,qubit=q,unitary=X) for q in zero_condition_qubits]
    mygate = ControlledU(n=n,target=target,unitary=unitary)
    circuit = X_gates + [mygate] + X_gates
    return circuit


# ---------------------------------------------------------------------------
# Stage 2-5: the decomposition pipeline (see cpp/src/Circuit.h)
# ---------------------------------------------------------------------------


def decompose_twolevel(tl: TwoLevel) -> Circuit:
    """Lower a TwoLevel to a Circuit (Nielsen-Chuang 4.5.2): walk a gray code so
    level0 becomes adjacent to level1, apply the controlled-U on that last
    transition, then undo the walk. Orient the 2x2 so a00 (level0's corner) sits on
    the target value the second-to-last code has.
    """
    # TODO: implement using gray_code, decompose_swap, controlled_circuit.
    n = num_qubits(tl.size)
    swaps = gray_code(tl)
    L = []
    for i in swaps[:-1]:
        L.extend(decompose_swap(i))
    last = swaps[-1]
    bit = (tl.level0 >> last.target) & 1
    if bit == 0:
        mat = tl.unitary
    else:
        X = np.array([[0,1],[1,0]],dtype=complex)
        mat = X @ tl.unitary @ X
    last_trans = controlled_circuit(n,last.target,last.control_vals,mat)
    L_inverse = list(reversed(L))
    circuit = L + [last_trans] + L_inverse
    return circuit


def decompose_controlled(n: int, controls: list[int], target: int, u: np.ndarray) -> Circuit:
    """Decompose C^k(U) (k = len(controls)) into singly-controlled gates C(U) and
    CNOTs (Nielsen-Chuang fig 4.8). Base cases: no control -> a plain SingleQubitGate;
    one control -> a CNOT if U == X else a CU. Otherwise, with V = sqrt(U):
        a. C(V) on target
        b. C^{k-1}(X) onto the pivot control
        c. C(V dagger) on target
        d. repeat b
        e. C^{k-1}(V) on target
    Phases are kept throughout.
    """
    # TODO: implement (recursive; use rotation.unitary2_sqrt for V).
    X = np.array([[0,1],[1,0]],dtype=complex)
    if len(controls) == 0:
        return [SingleQubitGate(n=n,qubit=target,unitary=u)]
    elif len(controls) == 1:
        if np.array_equal(u,X):
            return [CNOT(n=n,control=controls[0],target=target)]
        else:
            return[CU(n=n,control=controls[0],target=target,unitary=u)]
    else:
        v = rotation.unitary2_sqrt(u)
        controlled_V = CU(n=n,control=controls[-1],target=target,unitary=v)
        controlled_X = decompose_controlled(n,controls[:-1],controls[-1],X)
        controlled_V_dagger = CU(n=n,control=controls[-1],target=target,unitary=v.conj().T)
        controlled_V_big = decompose_controlled(n,controls[:-1],target,v)
        circuit = [controlled_V] + controlled_X + [controlled_V_dagger] + controlled_X + controlled_V_big
        return circuit


def decompose_controlledU(g: ControlledU) -> Circuit:
    """Lower a ControlledU (controlled on all other qubits) into CNOTs + C(U): build
    the list of all non-target qubits as controls and call decompose_controlled.
    """
    # TODO: implement.
    controls = [q for q in range(g.n) if q != g.target]
    return decompose_controlled(g.n,controls,g.target,g.unitary)


def decompose_cu(g: CU) -> Circuit:
    """Lower a singly-controlled C(U) into single-qubit gates + 2 CNOTs
    (Nielsen-Chuang Corollary 4.2 / fig 4.6). With U = e^{i alpha} A X B X C and
    A B C = I, emit: C, CNOT, B, CNOT, A on the target, plus a diag(1, e^{i alpha})
    phase on the control line. control=0: CNOTs vanish, target sees A B C = I;
    control=1: CNOTs act as X, target sees A X B X C = U with phase e^{i alpha}.
    """
    # TODO: implement using abc_decompose.
    myABC = abc_decompose(g.unitary)
    phase_mat = np.array([[1, 0], [0, np.exp(1j * d.alpha)]], dtype=complex)
    circuit = [SingleQubitGate(n=g.n,qubit=g.target,unitary=myABC.C),CNOT(n=g.n,control=g.control,target=g.target),SingleQubitGate(n=g.n,qubit=g.target,unitary=myABC.B),CNOT(n=g.n,control=g.control,target=g.target),SingleQubitGate(n=g.n,qubit=g.target,unitary=myABC.A),SingleQubitGate(n=g.n,qubit=g.control,unitary=myABC.phase_mat)]
    return circuit


def decompose_to_basis(u: np.ndarray) -> Circuit:
    """Fully lower a Unitary to a Circuit of only SingleQubitGate and CNOT, running
    the four stages in sequence:
        1. twolevel_decomposition: Unitary     -> TwoLevels
        2. decompose_twolevel:     TwoLevel    -> SingleQubitGate + ControlledU
        3. decompose_controlledU:  ControlledU -> CU + CNOT
        4. decompose_cu:           CU          -> SingleQubitGate + CNOT
    Each stage rewrites only its own gate type and passes the rest through unchanged.
    """
    # TODO: implement (run each rewrite pass over the circuit).
    step1 = twolevel_decomposition(u)
    step2 = []
    for i in step1:
        step2.extend(decompose_twolevel(i))
    step3 = []
    for i in step2:
        if isinstance(i,ControlledU):
            step3.extend(decompose_controlledU(i))
        else:
            step3.append(i)
    step4 = []
    for i in step3:
        if isinstance(i,CU):
            step4.extend(decompose_cu(i))
        else:
            step4.append(i)
    return step4


def ht_gates(n: int, qubit: int, word: str) -> Circuit:
    """Expand a flat H/T word into a Circuit of SingleQubitGate H/T gates on `qubit`.
    The word (leftmost char = leftmost matrix factor) is pushed in reverse so the
    circuit's application order (first gate first = rightmost factor) reproduces
    rotation.gates_to_unitary(word).
    """
    circuit = []
    for i in word[::-1]:
        if i == 'H':
            circuit.append(SingleQubitGate(n=n,qubit=qubit,unitary=rotation.H))
        else:
            circuit.append(SingleQubitGate(n=n,qubit=qubit,unitary=rotation.T))
    return circuit


def decompose_to_ht(u: np.ndarray, error: float) -> Circuit:
    """Fully lower a Unitary to a Circuit of only H, T, and CNOT gates (the discrete
    fault-tolerant basis): run decompose_to_basis, then replace each arbitrary
    SingleQubitGate with its {H, T} word from rotation.approximate_in_ht (CNOTs pass
    through). `error` is the per-gate angular tolerance (smaller -> longer, more
    accurate). Each word matches its gate up to a global phase; those per-gate phases
    factor out into one overall global phase, so the result reconstructs u up to
    global phase (compare with error_up_to_phase).
    """
    # TODO: implement using decompose_to_basis, ht_gates, and rotation.approximate_in_ht.
    basis_circuit = decompose_to_basis(u)
    new_circuit = []
    for i in basis_circuit:
        if isinstance(i,SingleQubitGate):
            new_circuit.extend(ht_gates(i.n,i.qubit,rotation.approximate_in_ht(i.unitary,error)))
        else:
            new_circuit.append(i)
    return new_circuit
