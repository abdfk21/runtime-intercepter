"""Empirical bootstrap confidence intervals calculation."""

from typing import Dict, List, Tuple
import numpy as np


class BootstrapCI:
    """Calculates non-parametric bootstrap confidence intervals."""

    @staticmethod
    def compute_ci(
        data: List[float],
        iterations: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, float]:
        if not data:
            return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        arr = np.array(data, dtype=float)
        n = len(arr)
        if n == 1:
            val = float(arr[0])
            return {"mean": val, "std": 0.0, "ci_lower": val, "ci_upper": val}

        rng = np.random.RandomState(seed)
        boot_means = []
        for _ in range(iterations):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means.append(np.mean(sample))

        lower_pct = 100.0 * (alpha / 2.0)
        upper_pct = 100.0 * (1.0 - alpha / 2.0)

        ci_lower = float(np.percentile(boot_means, lower_pct))
        ci_upper = float(np.percentile(boot_means, upper_pct))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))

        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "alpha": alpha,
            "iterations": iterations,
        }
