from typing import Optional
import torch


def slow_adaptive_cross_approximation(
    A: torch.Tensor, max_iter: int = 200, verbose: bool = False,
) -> list[int]:

    n, _ = A.shape

    # Empty list of indeces
    inducing_points = []

    # Residual matrix
    R = A

    # Iteration counter
    iter = 0

    while iter < max_iter:
        # Verbose state
        if verbose:
            print(f"Iter {iter:03d}")

        # Select best point
        i = torch.argmax(torch.abs(R.diag()))

        # Update residual
        R = R - torch.outer(
            R[:, i] / R[i, i],
            R[i, :])

        # Append example
        inducing_points.append(i)

        # Update iteration
        iter += 1

    return inducing_points


def adaptive_cross_approximation(
    A: torch.Tensor, max_iter: int = 200,
    precision: Optional[float] = None, verbose: bool = False,
    debug: bool = False
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

    # Iteration counter
    iter = 0

    while iter < max_iter:
        # Verbose state
        if verbose:
            print(f"Iter {iter:03d}")

        # Select best point
        i = torch.argmax(torch.abs(diag))

        # Check correctness of the index
        if debug:
            assert torch.allclose(i, torch.argmax(torch.abs(R.diag())))

        # Update partial residual
        rho[iter] = A[:, i]
        acc = 0.
        for j in range(iter):
            acc += rho[j, i] * rho[j] / alpha[j]
        rho[iter] -= acc
        # Adjust precision on the residual
        if precision is not None:
            rho[iter][torch.abs(rho[iter]) < precision] = 0
        alpha[iter] = rho[iter, i]

        # Check correctness of the residual
        if debug:
            if verbose:
                print("rho:", rho[iter])
                print("alpha:", alpha[iter])
                print("rho_gt:", R[:, i])
                print("alpha_gt:", R[i, i])
            assert torch.allclose(rho[iter], R[:, i])
            assert torch.allclose(alpha[iter], R[i, i])

        # Update diagonal
        for j in range(n):
            diag[j] = diag[j] - torch.square(rho[iter, j]) / alpha[iter]

        # Adjust precision for the diagonal
        if precision is not None:
            diag[torch.abs(diag) < precision] = 0

        # Update residual (explicit)
        if debug:
            R = R - torch.outer(
                R[:, i] / R[i, i],
                R[i, :])

        # Check correctness of the diagonal
        if debug:
            if verbose:
                print("diag:", diag)
                print("diag_gt:", R.diag())
            assert torch.allclose(diag, R.diag())

        # Append example
        inducing_points.append(i)

        # Update iteration
        iter += 1

    return inducing_points


def main(n: int = 10000, m: int = 10):
    # Used to measure executions
    import time

    # Generate data
    A = torch.randn(n, n)
    A = A.t() @ A

    # Compute ACA (Fast)
    time_a = time.time()
    inducing_points_a = adaptive_cross_approximation(
        A, max_iter=m
    )
    time_b = time.time()
    print("Done fast")

    # Compute ACA (Slow)
    time_c = time.time()
    inducing_points_b = slow_adaptive_cross_approximation(
        A, max_iter=m
    )
    time_d = time.time()
    print("Done slow")

    # Set of indices
    inducing_points_a = sorted(list(set(inducing_points_a)))
    inducing_points_b = sorted(list(set(inducing_points_b)))

    # Print results
    print(f"Inducing points (Fast) in {time_b-time_a:.2f}:"
          f"{inducing_points_a}")
    print(f"Inducing points (Slow) in {time_d-time_c:.2f}:"
          f"{inducing_points_b}")


if __name__ == "__main__":
    main()
