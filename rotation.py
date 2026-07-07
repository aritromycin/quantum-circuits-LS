import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)
X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle

def R_form(theta: float, n: np.ndarray) -> np.ndarray:
    I = np.array([[1,0],[0,1]],dtype=DTYPE)
    return I*np.cos(theta/2) - (n[0]*X + n[1]*Y + n[2]*Z)*(np.sin(theta/2)*1j)
    


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    myDet = np.linalg.det(g)
    alpha = np.angle(myDet)/2
    g_prime = np.exp(-1j*alpha)*g
    theta = 2 * np.arccos(np.trace(g_prime).real / 2)
    n_x = (((1j/2)*np.trace(X @ g_prime))/(np.sin(theta/2))).real
    n_y = (((1j/2)*np.trace(Y @ g_prime))/(np.sin(theta/2))).real
    n_z = (((1j/2)*np.trace(Z @ g_prime))/(np.sin(theta/2))).real
    b = Bloch()
    b.alpha = alpha
    b.n = np.array([n_x, n_y, n_z])
    b.theta = theta
    return b


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
n1 = np.array([-(1/np.tan(np.pi/8)), 1, 1/np.tan(np.pi/8)])/np.sqrt(2*(1/np.tan(np.pi/8)**2) + 1) 
n2 = np.array([INV_SQRT2, np.sqrt(2)*np.tan(np.pi/8), -INV_SQRT2])/np.sqrt(2/((np.tan(np.pi/8)**2)) + 1)

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)
        
    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    # TODO(student): implement using the steps above.
    gamma_plus_alpha = np.arctan2(-np.dot(b.n,a1)*np.sin(b.theta), np.cos(b.theta))
    beta = np.arccos(np.clip(np.cos(b.theta)/np.cos(gamma_plus_alpha), -1, 1))
    if np.isclose(np.sin(beta), 0):
        gamma = -gamma_plus_alpha
        alpha = 0.0
    else:
        gamma_minus_alpha = np.arccos(np.clip(-np.dot(b.n, a2)*np.sin(b.theta)/np.sin(beta), -1, 1))
        gamma = (gamma_plus_alpha + gamma_minus_alpha) / 2
        alpha = (gamma_plus_alpha - gamma_minus_alpha) / 2
    R_alpha = R_form(alpha,a1)
    R_beta = R_form(beta,a2)
    R_gamma = R_form(gamma,a1)
    # Correct matrix tracking sequence for global phase calculation:
    combined_rotations = R_alpha @ R_beta @ R_gamma
    global_phase = np.angle((np.exp(1j * b.alpha) * (R_form(b.theta, b.n) @ np.linalg.inv(combined_rotations)))[0][0])
    return (alpha,beta,gamma,global_phase)

def zyz_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Same as n1n2n1_angles, but factoring into the standard Z-Y-Z frame instead
    of the H/T frame: u = e^{i global_phase} * Rz(alpha) * Ry(beta) * Rz(gamma).
    """
    a1 = np.array([0, 0, 1])  # Z axis
    a2 = np.array([0, 1, 0])  # Y axis

    gamma_plus_alpha = np.arctan2(-np.dot(b.n, a1) * np.sin(b.theta), np.cos(b.theta))
    beta = np.arccos(np.clip(np.cos(b.theta) / np.cos(gamma_plus_alpha), -1, 1))
    if np.isclose(np.sin(beta), 0):
        gamma = -gamma_plus_alpha
        alpha = 0.0
    else:
        gamma_minus_alpha = np.arccos(np.clip(-np.dot(b.n, a2) * np.sin(b.theta) / np.sin(beta), -1, 1))
        gamma = (gamma_plus_alpha + gamma_minus_alpha) / 2
        alpha = (gamma_plus_alpha - gamma_minus_alpha) / 2

    R_alpha = R_form(alpha, a1)
    R_beta = R_form(beta, a2)
    R_gamma = R_form(gamma, a1)
    combined_rotations = R_alpha @ R_beta @ R_gamma
    global_phase = np.angle(
        (np.exp(1j * b.alpha) * (R_form(b.theta, b.n) @ np.linalg.inv(combined_rotations)))[0][0]
    )
    return (alpha, beta, gamma, global_phase)


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    # TODO(student): implement using the hint above.
    wrap = angle % TWO_PI
    i = 1
    while True:
        wrapchecker = (i*LAMBDA_PI) % TWO_PI
        if min(abs(wrapchecker-wrap), TWO_PI - abs(wrapchecker-wrap)) < tolerance:
            return i
        i=i+1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """
    # TODO(student): implement using the steps above.
    bloch_u = to_bloch(u)
    alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
    k = approx_angle_with_tolerance(alpha, tolerance) 
    l = approx_angle_with_tolerance(beta,  tolerance)   
    m = approx_angle_with_tolerance(gamma, tolerance)  
    return (k,l,m)

def unitary2_sqrt(u:np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(u)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    v = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T
    return v


# ---------------------------------------------------------------------------
# H/T word machinery for approximating a 2x2 unitary in {H, T} (see cpp/src/HT.h).
#
# M1, M2 are short H/T words that realize rotations by THETA_M = 2*LAMBDA_PI about
# the axes a1, a2. A word is a flat string of 'H'/'T' characters, read left-to-right
# as a matrix product (leftmost char = leftmost/outermost factor).
# ---------------------------------------------------------------------------

# alternating (T-power, H-power, ...) exponents, starting with T
M1_WORD = [7, 1, 1, 1]
M2_WORD = [2, 1, 1, 1, 6, 1, 7, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 7, 1, 6]


def expand_word(word: list[int]) -> str:
    """Flatten an alternating (T-power, H-power, ...) exponent list into a literal
    string of 'H'/'T' gates (left-to-right). Even indices are T, odd indices are H.
    """
    # TODO: implement.
    my_word = ""
    for i in range(len(word)):
        if i%2 == 0:
            for j in range(word[i]):
                my_word = my_word + 'T'
        else:
            for j in range(word[i]):
                my_word = my_word + 'H'
    return my_word


# flat H/T strings for the two building-block words (computed once expand_word works)
M1_STR = expand_word(M1_WORD)
M2_STR = expand_word(M2_WORD)


def gates_to_unitary(gates: str) -> np.ndarray:
    """The 2x2 unitary of a flat H/T gate string (left-to-right product)."""
    # TODO: implement (multiply H / T for each char, starting from I).
    mat = np.eye(2,dtype=DTYPE)
    for i in gates:
        if i == 'H':
            mat = mat @ H
        else:
            mat = mat @ T
    return mat


def invert_gates(gates: str) -> str:
    """Inverse of a flat H/T word: reverse the gate order and invert each gate.
    H^-1 = H; the {H, T} basis has no T-dagger, so T^-1 must be spelled as T^7.
    """
    # TODO: implement.
    inv = ""
    for i in gates[::-1]:
        if i == 'H':
            inv = inv + 'H'
        else:
            inv = inv + 'TTTTTTT'
    return inv


def power_gates(base: str, k: int) -> str:
    """The k-th power of a flat H/T word: base repeated k times. Negative k uses the
    inverse word (invert_gates).
    """
    # TODO: implement.
    if k > 0:
        return base*k
    elif k < 0:
        return invert_gates(base)*(-k)
    else:
        return ""


def approximate_in_ht(u: np.ndarray, error: float) -> str:
    """Approximate a 2x2 unitary `u` by a flat H/T word (up to global phase) to the
    angular tolerance `error` (smaller -> longer, more accurate).

    Use decompose_2x2 to get the powers (k, l, m) with u ~= M1^k M2^l M1^m, then
    assemble the word:

        power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m).
    """
    # TODO: implement using decompose_2x2 and power_gates.
    k,l,m = decompose_2x2(u,error)
    approx = power_gates(M1_STR, k) + power_gates(M2_STR, l) + power_gates(M1_STR, m)
    return approx
