from __future__ import annotations

from typing import Any, Optional

import numpy as np

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class PBIL:
    def __init__(
        self,
        cat_num: np.ndarray,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        n_update_from: Optional[int] = None,
    ):

        self._n_ca = len(cat_num)
        self._n = self._n_ca
        assert self._n_ca != 2, (
            "This implementation does not support other than 2 categories"
        )
        assert np.all(cat_num > 1), "The number of categories must be larger than 1"

        self._popsize = population_size
        self._q = np.full((self._n_ca, 1), 0.5)

        self._g = 0
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        self.LR = 1 / self._n  # Learning rate
        self.n_update_from = n_update_from if n_update_from is not None else 2

    def __setstate__(self, state: dict[str, Any]) -> None:
        state["_C"] = _decompress_symmetric(state["_c_1d"])
        del state["_c_1d"]
        self.__dict__.update(state)
        # Set _rng for unpickled object.
        setattr(self, "_rng", np.random.RandomState())

    @property
    def cat_dim(self) -> int:
        """A number of dimensions of categorical variable"""
        return self._n_ca

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n

    @property
    def cat_num(self) -> np.ndarray:
        """Numbers of categories"""
        return self._K

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)

    def ask(self) -> tuple[np.ndarray]:
        """Sample a parameter"""
        c = self._sample_solution()
        return c

    def _sample_solution(self) -> tuple[np.ndarray]:
        rand_q = self._rng.rand(self._n_ca, 1)
        tmp = np.hstack([1 - self._q, self._q])
        cum_q = tmp.cumsum(axis=1)
        c = (cum_q - tmp <= rand_q) & (rand_q < cum_q)
        return c

    def tell(
        self, solutions: list[tuple[tuple[np.ndarray, np.ndarray], float]]
    ) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."

        self._g += 1
        solutions.sort(key=lambda s: s[1], reverse=True)
        self.best_solution_q = solutions[0][0][:, 1:2]
        update_from_solutions = solutions[: self.n_update_from]

        self.avg_fvalue = np.mean([s[1] for s in solutions])

        # Update of categorical distribution
        c_update_from = np.array(
            [sol[0][:, 1:2].flatten() for sol in update_from_solutions]
        )

        self._q = (
            self._q * (1.0 - self.LR)
            + np.mean(c_update_from, axis=0).reshape(self._n_ca, 1) * self.LR
        )


def _decompress_symmetric(sym1d: np.ndarray) -> np.ndarray:
    n = int(np.sqrt(sym1d.size * 2))
    assert (n * (n + 1)) // 2 == sym1d.size
    R, C = np.triu_indices(n)
    out = np.zeros((n, n), dtype=sym1d.dtype)
    out[R, C] = sym1d
    out[C, R] = sym1d
    return out
