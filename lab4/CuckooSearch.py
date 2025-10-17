import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def rastrigin(x):
    A = 10.0
    n = x.size
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def ensure_bounds(vec, bounds):
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return np.minimum(np.maximum(vec, lower), upper)


def levy_flight(Lambda, dim):
    sigma_u = (gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1.0, size=dim)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step


def cuckoo_search(obj_func, dim, bounds, n_nests=25, pa=0.25, alpha=0.01, max_iter=500, verbose=True):
    nests = np.array([[np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)] for _ in range(n_nests)])
    fitness = np.array([obj_func(nests[i]) for i in range(n_nests)])

    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()
    best_score = fitness[best_idx]

    history = [best_score]

    Lambda = 1.5

    for t in range(max_iter):
        for i in range(n_nests):
            s = nests[i].copy()
            step = levy_flight(Lambda, dim)
            stepsize = alpha * step * (s - best)
            new_s = s + stepsize * np.random.randn(dim)
            new_s = ensure_bounds(new_s, bounds)

            new_f = obj_func(new_s)
            if new_f < fitness[i]:
                nests[i] = new_s
                fitness[i] = new_f
                if new_f < best_score:
                    best_score = new_f
                    best = new_s.copy()

        K = int(np.ceil(pa * n_nests))
        if K > 0:
            worst_idx = np.argsort(fitness)[-K:]
            for idx in worst_idx:
                nests[idx] = np.array([np.random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)])
                fitness[idx] = obj_func(nests[idx])
                if fitness[idx] < best_score:
                    best_score = fitness[idx]
                    best = nests[idx].copy()

        history.append(best_score)

        if verbose and (t % 50 == 0 or t == max_iter - 1):
            print(f"Iter {t:4d} | best = {best_score:.6e}")

    return best, best_score, history


# Demo usage
if __name__ == "__main__":
    np.random.seed(42)
    dim = 10
    bounds = [(-5.12, 5.12)] * dim
    n_nests = 40
    pa = 0.25
    alpha = 0.01
    max_iter = 1000

    best_x, best_val, history = cuckoo_search(
        obj_func=rastrigin,
        dim=dim,
        bounds=bounds,
        n_nests=n_nests,
        pa=pa,
        alpha=alpha,
        max_iter=max_iter,
        verbose=True,
    )

    print("\nFinal result:")
    print("Best value:", best_val)
    print("Best position:", np.round(best_x, 6))

    plt.plot(history)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best objective (log scale)")
    plt.title(f"Cuckoo Search on Rastrigin (dim={dim})")
    plt.grid(True)
    plt.show()
