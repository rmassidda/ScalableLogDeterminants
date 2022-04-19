import torch


def adaptive_cross_approximation(
    A: torch.Tensor, eps: float = 10, max_iter: int = 200,
    verbose: bool = False
) -> list[int]:

    # Empty list of indeces
    inducing_points = []

    # Residual matrix
    R = A

    # Check stop criterion
    converged = False  # np.trace(R) <= eps

    # Iteration counter
    iter = 0

    # Verbose
    # if verbose:
    #     print(f"Iter {iter:03d} | Trace: {np.trace(R):.3f}")

    while not converged and iter < max_iter:
        i = torch.argmax(torch.abs(R.diag()))

        # Update residual
        R = R - torch.outer(
            R[:, i] / R[i, i],
            R[i, :])

        # Append example
        inducing_points.append(i)

        # Verbose state
        if verbose:
            print(f"Iter {iter+1:03d}")#:#: | Trace: {np.trace(R):.3f}")

        # Update convergence
        converged = False  # np.trace(R) <= eps

        # Update iteration
        iter += 1

    return inducing_points
