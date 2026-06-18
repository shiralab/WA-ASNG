from __future__ import annotations

import math
from typing import Optional

import numpy as np


class ASNG:
    def __init__(
        self,
        cat_num: np.ndarray,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cat_param: Optional[np.ndarray] = None,
    ):

        self._n_ca = len(cat_num)
        self._n = self._n_ca
        assert self._n_ca > 0, "The dimension of categorical variable must be positive"
        assert np.all(cat_num > 1), "The number of categories must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(self._n))
        assert population_size > 0, "popsize must be non-zero positive value."

        mu = math.ceil(population_size / 4)

        weights = np.zeros(population_size)
        weights[:mu] = 1.0
        weights[-mu:] = -1.0

        self._popsize = population_size
        self._mu = mu

        self._weights = weights

        # categorical distribution
        # Parameters in categorical distribution with fewer categories
        # must be zero-padded at the end.
        self._K = cat_num
        self._Kmax = np.max(self._K)
        if cat_param is None:
            self._q = np.zeros((self._n_ca, self._Kmax))
            for i in range(self._n_ca):
                self._q[i, : self._K[i]] = 1 / self._K[i]
        else:
            assert cat_param.shape == (
                self._n_ca,
                self._Kmax,
            ), "Invalid shape of categorical distribution parameter"
            for i in range(self._n_ca):
                assert np.all(cat_param[i, self._K[i] :] == 0), (
                    "Parameters in categorical distribution with fewer categories "
                    "must be zero-padded at the end"
                )
            assert np.all((cat_param >= 0) & (cat_param <= 1)), (
                "All elements in categorical distribution parameter must be between 0 and 1"
            )
            assert np.allclose(np.sum(cat_param, axis=1), 1), (
                "Each row in categorical distribution parameter must sum to 1"
            )
            self._q = cat_param

        self._q_min = 1 / (self._n_ca * (self._K - 1))

        # ASNG
        self._param_sum = np.sum(cat_num - 1)
        self._alpha = 1.5
        self._delta_init = 1.0
        self._Delta = 1.0
        self._Delta_max = np.inf
        self._gamma = 0.0
        self._s = np.zeros(self._param_sum)
        self._delta = self._delta_init / self._Delta
        self._eps = self._delta

        self._g = 0
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)

        self.best_solution_q = None

        self.Eq = np.zeros((self._n_ca, self._Kmax))
        self.Vq = 0.0
        self._beta_q = 0.05
        self.hat_SNRq = 0.0

        self.ngrad_norm = 0.0

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
        c = self._sample_solution()
        return c

    def _sample_solution(self) -> tuple[np.ndarray]:
        rand_q = self._rng.rand(self._n_ca, 1)
        cum_q = self._q.cumsum(axis=1)
        c = (cum_q - self._q <= rand_q) & (rand_q < cum_q)
        return c

    def tell(
        self, solutions: list[tuple[tuple[np.ndarray, np.ndarray], float]]
    ) -> None:
        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."

        self._g += 1
        solutions.sort(key=lambda s: s[1], reverse=True)
        self.best_solution_q = solutions[0][0]

        self.avg_fvalue = np.mean([s[1] for s in solutions])

        self.old_delta = self._delta

        c = np.array([s[0] for s in solutions])
        ngrad = (self._weights[:, np.newaxis, np.newaxis] * (c - self._q)).sum(axis=0)
        self.ngrad_norm = np.linalg.norm(ngrad)

        sl = []
        for i, K in enumerate(self._K):
            q_i = self._q[i, : K - 1]
            q_i_K = self._q[i, K - 1]
            s_i = 1.0 / np.sqrt(q_i) * ngrad[i, : K - 1]
            s_i += np.sqrt(q_i) * ngrad[i, : K - 1].sum() / (q_i_K + np.sqrt(q_i_K))
            sl += list(s_i)
        ngrad_sqF = np.array(sl)

        pnorm = np.sqrt(np.dot(ngrad_sqF, ngrad_sqF)) + 1e-30
        self.pnorm = pnorm
        self._eps = self._delta / pnorm
        self._q += self._eps * ngrad

        # Update of ASNG
        self._delta = self._delta_init / self._Delta
        beta = self._delta / (self._param_sum**0.5)

        # Margin Correction (Originally in ASNG)
        for i in range(self._n_ca):
            Ki = self._K[i]
            self._q[i, :Ki] = np.maximum(self._q[i, :Ki], self._q_min[i])
            q_sum = self._q[i, :Ki].sum()
            tmp = q_sum - self._q_min[i] * Ki
            self._q[i, :Ki] -= (q_sum - 1) * (self._q[i, :Ki] - self._q_min[i]) / tmp
            self._q[i, :Ki] /= self._q[i, :Ki].sum()

        self._s = (1 - beta) * self._s + np.sqrt(beta * (2 - beta)) * ngrad_sqF / pnorm
        self._gamma = (1 - beta) ** 2 * self._gamma + beta * (2 - beta)
        self._Delta *= np.exp(
            beta * (self._gamma - np.dot(self._s, self._s) / self._alpha)
        )
        self._Delta = min(self._Delta, self._Delta_max)
        self._Delta = max(self._Delta, 1 / (2 * (self._param_sum**0.5)))

        self.hat_SNRq = (np.linalg.norm(self._s)) ** 2 / self._gamma
        self.hat_signal = (np.linalg.norm(self._s)) ** 2 - self._gamma
        self.delta_change_ratio = (self._delta - self.old_delta) / self.old_delta
        self.entropy = -np.sum(
            self._q[:, 0] * np.log2(self._q[:, 0])
            + self._q[:, 1] * np.log2(self._q[:, 1])
        )
