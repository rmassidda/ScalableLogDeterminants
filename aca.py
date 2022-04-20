import torch


def adaptive_cross_approximation(
    A: torch.Tensor, eps: float = 10, max_iter: int = 200,
    verbose: bool = False
) -> list[int]:

    n, _ = A.shape

    # Empty list of indeces
    inducing_points = []

    # Residual matrix
    R = A

    # Partial residual matrix
    rho = torch.zeros((max_iter, n))
    alpha = torch.zeros((max_iter,))
    diag = R.diag()

    # Check stop criterion
    converged = False  # np.trace(R) <= eps

    # Iteration counter
    iter = 0

    while not converged and iter < max_iter:
        # Verbose state
        if verbose:
            print(f"Iter {iter:03d}")#:#: | Trace: {np.trace(R):.3f}")

        # Select best point
        i = torch.argmax(torch.abs(diag))

        # Check correctness of the index
        assert torch.allclose(i, torch.argmax(torch.abs(R.diag())))

        # Update partial residual
        rho[iter] = A[:, i]
        for j in range(iter):
            rho[iter] = rho[iter] - rho[j, i] * rho[j] / alpha[j]
        alpha[iter] = rho[iter, i]

        # Check correctness of the residual
        print("rho:", rho[iter])
        print("alpha:", alpha[iter])
        print("rho_gt:", R[:, i])
        print("alpha_gt:", R[i, i])
        assert torch.allclose(rho[iter], R[:, i])
        assert torch.allclose(alpha[iter], R[i, i])

        # Update diagonal
        for j in range(n):
            diag[j] = diag[j] - torch.square(rho[iter, j]) / alpha[iter]

        # Update residual (explicit)
        R = R - torch.outer(
            R[:, i] / R[i, i],
            R[i, :])

        # Check correctness of the diagonal
        print("diag:", diag)
        print("diag_gt:", R.diag())
        assert torch.allclose(diag, R.diag())

        # Append example
        inducing_points.append(i)

        # Update convergence
        converged = False  # np.trace(R) <= eps

        # Update iteration
        iter += 1

    return inducing_points


def main(n: int = 5, m: int = 4):
    # Generate data
    A = torch.randn(n, n)
    A = A.t() @ A

    # Compute ACA
    inducing_points = adaptive_cross_approximation(A, max_iter=3, verbose=True)

    # Print results
    print(f"Inducing points: {inducing_points}")


if __name__ == "__main__":
    main()