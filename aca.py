import numpy as np


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def adaptive_cross_approximation(
    A: np.ndarray, eps: float = 10, max_iter: int = 200,
    verbose: bool = False
) -> list[int]:

    # Empty list of indeces
    inducing_points = []

    # Residual matrix
    R = A

    # Check stop criterion
    converged = np.trace(R) <= eps

    # Iteration counter
    iter = 0

    # Verbose
    if verbose:
        print(f"Iter {iter:03d} | Trace: {np.trace(R):.3f}")

    while not converged and iter < max_iter:
        i = np.argmax(np.abs(R.diagonal()))

        # Update residual
        R = R - np.outer(
            R[:, i] / R[i, i],
            R[i, :])

        # Append example
        inducing_points.append(i)

        # Verbose state
        if verbose:
            print(f"Iter {iter+1:03d} | Trace: {np.trace(R):.3f}")

        # Update convergence
        converged = np.trace(R) <= eps

        # Update iteration
        iter += 1

    return inducing_points


def fast_aca_spsd(A: np.ndarray, eps: float = 10, max_iter: int = 200,
                  verbose: bool = False) -> list[int]:

    U, V = adaptive_cross_approximation(
        A, eps, max_iter, stop_criterion=StopCriterion.TRACE,
        partial_pivoting=False, spsd=True, return_indeces=True,
        verbose=verbose)

    # Assert that the ACA produced two lists of ints
    assert isinstance(U, list)
    assert isinstance(V, list)

    # Check that the lists are equal
    assert U == V

    return U


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
