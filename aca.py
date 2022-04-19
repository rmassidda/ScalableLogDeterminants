from enum import Enum, auto
from typing import Union
import numpy as np


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class StopCriterion(Enum):
    CHEBYSHEV = auto()
    DOTPROD = auto()
    TRACE = auto()


def adaptive_cross_approximation(
    A: np.ndarray, eps: float = 10, max_iter: int = 200,
    stop_criterion: StopCriterion = StopCriterion.CHEBYSHEV,
    partial_pivoting: bool = False, spsd: bool = False,
    return_indeces: bool = False,
    verbose: bool = False
) -> Union[tuple[np.ndarray, np.ndarray], tuple[list[int], list[int]]]:

    # Shape
    n, m = A.shape

    # Empty lists
    U = []
    V = []

    # Empty list of indeces
    U_indeces = []
    V_indeces = []

    # Residual matrix
    R = A

    # Index of a random row
    fixed_row = np.random.randint(0, n)
    # Mask for already visited columns
    mask_row = np.ones(n, dtype=bool)
    mask_col = np.ones(m, dtype=bool)

    # Check stop criterion
    if stop_criterion == StopCriterion.CHEBYSHEV:
        converged = np.linalg.norm(R, ord=np.inf) <= eps
    elif stop_criterion == StopCriterion.DOTPROD:
        converged = False
    elif stop_criterion == StopCriterion.TRACE:
        converged = np.trace(R) <= eps

    if verbose:
        print("R_0")
        print(R)
        print(f"Trace: {np.trace(R)}")

    # Start iteration
    iter = 0
    while not converged and iter < max_iter:
        if partial_pivoting:
            i = fixed_row
            j = np.argmax(np.abs(R[i] * mask_row))
        elif spsd:
            i = np.argmax(np.abs(R.diagonal()))
            j = i
            assert R[i, j] == np.max(np.abs(R))
        else:
            # Largest value in the matrix
            # NOTE: in NumPy the index concerns the position within
            #       the actual position in the array. This might not
            #       coincide with the position within the matrix.
            #       Given a shape, the function unravel_index returns
            #       the position within the matrix.
            i, j = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            i = int(i)
            j = int(j)

        # Check that the index is not already visited
        if not mask_row[i] or not mask_col[j]:
            print("Index already visited")
            converged = True
            continue

        # Mark the removed indices
        mask_row[i] = False
        mask_col[j] = False

        # Insert the column
        U.append(R[:, j] / R[i, j])
        U_indeces.append(j)

        # Insert the row
        V.append(R[i])
        V_indeces.append(i)

        # Update residual
        # NOTE: could I do this on the fly?
        R = R - np.outer(U[-1], V[-1])

        # Verbose state
        if verbose:
            print(f"Removing {i,j}")
            print(f"Outer U_{iter} V_{iter}")
            print(np.outer(U[-1], V[-1]))
            print(f"R_{iter + 1}")
            print(R)
            print(f"Trace: {np.trace(R)}")

        # Update the fixed row
        if partial_pivoting:
            # Maximum entry in u that has not been selected
            fixed_row = np.argmax(np.abs(U[-1] * mask_row))

        # Update stop criterion
        if stop_criterion == StopCriterion.CHEBYSHEV:
            converged = np.linalg.norm(R, ord=np.inf) <= eps
        elif stop_criterion == StopCriterion.DOTPROD:
            converged = np.dot(np.square(U[-1]), np.square(V[-1])) <= eps
        elif stop_criterion == StopCriterion.TRACE:
            converged = np.trace(R) <= eps

        if converged:
            print('Converged')

        # Update iteration
        iter += 1

        # Check if the limit has been reached
        if iter == max_iter:
            print(f"Maximum number of iterations reached: {iter}")

    if return_indeces:
        return U_indeces, V_indeces
    else:
        U = np.column_stack(U)
        V = np.row_stack(V)
        return U, V


def main(n: int = 2000, m: int = 3):
    # Observations matrix of n
    # examples with dimension m
    A = np.random.normal(0., 1., (n, m))

    # Squared distance kernel
    def kernel(x, y):
        return np.exp(-0.5 * np.sum(np.square(x - y)))

    # Compute the kernel matrix
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(A[i], A[j])

    # Diagonal correction
    K = K + np.eye(n) * 1e-6
    assert np.all(np.linalg.eigvals(K) > 0)
    print(f"K: {K.shape}")

    U, V = adaptive_cross_approximation(
        K, eps=5e-1, spsd=True,
        stop_criterion=StopCriterion.TRACE)

    # Make the typechecker happy
    assert isinstance(U, np.ndarray) and isinstance(V, np.ndarray)

    print(f"U: {U.shape}")
    print(f"V: {V.shape}")

    # Low rank approximation
    K_lowrank = np.dot(U, V)

    # Trace of the error
    print(f"Trace error: {np.trace(K - K_lowrank)}")

    # Frobenius Norm
    print(f"Frobenius Norm: {np.linalg.norm(K - K_lowrank, ord='fro')}")


if __name__ == "__main__":
    main()
