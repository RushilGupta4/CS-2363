import numpy as np


def householder_vector(x):
    v = x.copy()
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return np.zeros_like(x)
    v[0] += np.sign(x[0]) * norm_x
    v = v / np.linalg.norm(v)
    return v


def qr_factorization(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for j in range(min(m, n)):
        x = R[j:, j]
        if np.allclose(x, 0):
            continue
        v = householder_vector(x)

        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])
        Q[:, j:] -= 2.0 * np.outer(Q[:, j:] @ v, v)
    return Q, R


def qr_algorithm(A, max_iter=100, tol=1e-10):
    n = A.shape[0]
    Q_acc = np.eye(n)
    A_k = A.copy()

    for _ in range(max_iter):
        Q, R = qr_factorization(A_k)
        A_k = R @ Q
        Q_acc = Q_acc @ Q

        A_k = (A_k + A_k.T) / 2

        off_diagonal = A_k - np.diag(np.diagonal(A_k))
        if np.all(np.abs(off_diagonal) < tol):
            break

    eigenvalues = np.diag(A_k)
    eigenvectors = Q_acc

    return eigenvalues, eigenvectors


def svd(A):
    A = A.astype(float)
    m, n = A.shape

    # STEPS FROM Trefethen Bau

    # 1) Compute A^T A
    ATA = np.dot(A.T, A)

    # 2) Find eigenvalues and eigenvectors of A^T A
    eigenvalues, V = qr_algorithm(ATA)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]

    # 3) Compute Sigma (square root of eigenvalues), truncate all < 1e-10
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))
    singular_values = singular_values[singular_values > 1e-10]
    k = len(singular_values)

    # Create k×k Sigma matrix (instead of m×m)
    Sigma = np.diag(singular_values)

    # Truncate V to only keep k columns
    V = V[:, :k]

    # 4) Solve U @ Sigma = A @ V
    AV = A @ V

    # Create skinny U matrix (m×k instead of m×n)
    U = np.zeros((m, k))
    for i in range(k):
        U[:, i] = AV[:, i] / singular_values[i]

    return U, Sigma, V.T


def main(d=3):
    # Creating our matrix A
    m = n = d
    A = np.random.rand(m, n)

    from time import perf_counter_ns

    start = perf_counter_ns()
    U, Sigma, Vt = svd(A)
    end = perf_counter_ns()
    t1 = (end - start) / 1e9
    print(f"Time taken: {t1} seconds")

    start = perf_counter_ns()
    Unp, Sigmanp, Vtnp = np.linalg.svd(A)
    Sigmanp = np.diag(Sigmanp)
    end = perf_counter_ns()
    t2 = (end - start) / 1e9
    print(f"Time taken: {t2} seconds")

    A_reconstructed = U @ Sigma @ Vt 

    print(f"Speedup: {round(t1 / t2, 3)}x")

    print(np.linalg.norm(A - A_reconstructed))
    print(sum(sum(abs(Sigma - Sigmanp))) / d)


def timeit(n=10):
    from time import perf_counter_ns

    times = []
    d = 50
    A = np.random.rand(d, d)
    for _ in range(n):
        start = perf_counter_ns()
        svd(A)
        end = perf_counter_ns()
        times.append((end - start) / 1e9)

    print(sum(times), sum(times) / n)


if __name__ == "__main__":
    # timeit()
    main(1000)
