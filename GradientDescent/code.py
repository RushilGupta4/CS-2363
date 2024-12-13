import numpy as np
import matplotlib.pyplot as plt

# Given matrices and initial guess
A = np.array([[4, 1], [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
x0 = np.array([2, 1], dtype=float)


def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x


def grad_f(x):
    return A @ x - b


# Steepest Descent Implementation
def steepest_descent(A, b, x0, tol=1e-8, max_iter=1000):
    x = x0.copy()
    f_values = []
    for i in range(max_iter):
        r = grad_f(x)  # gradient
        f_values.append(f(x))
        if np.linalg.norm(r) < tol:
            break
        alpha = (r @ r) / (r @ (A @ r))
        x = x - alpha * r
    return x, f_values, i + 1


# Conjugate Gradient Implementation
def conjugate_gradient(A, b, x0, tol=1e-8, max_iter=1000):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    f_values = []
    for i in range(max_iter):
        f_values.append(f(x))
        if np.linalg.norm(r) < tol:
            break
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            r = r_new
            f_values.append(f(x))
            break
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new
    return x, f_values, i + 1


# Run the methods
x_sd, f_sd, it_sd = steepest_descent(A, b, x0)
x_cg, f_cg, it_cg = conjugate_gradient(A, b, x0)

# Compute final function values
f_min_sd = f(x_sd)
f_min_cg = f(x_cg)

# Print results
print("Steepest Descent:")
print("  Optimal x =", x_sd)
print("  f(x*) =", f_min_sd)
print("  Iterations =", it_sd)

print("\nConjugate Gradient:")
print("  Optimal x =", x_cg)
print("  f(x*) =", f_min_cg)
print("  Iterations =", it_cg)

# Plot the convergence
plt.figure(figsize=(8, 6))
plt.plot(f_sd, label="Steepest Descent")
plt.plot(f_cg, label="Conjugate Gradient")
plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.title("Convergence of f(x) for Steepest Descent vs Conjugate Gradient")
plt.legend()
plt.grid(True)
plt.savefig("Convergence.png")
