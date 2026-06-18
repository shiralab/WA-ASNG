from __future__ import annotations

import math
from typing import Optional

# import cast
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

torch.set_num_threads(1)


class WAASNG:
    def __init__(
        self,
        cat_num: np.ndarray,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        cat_param: Optional[np.ndarray] = None,
        n_epochs: int = 30,
        alpha: float = 1.5,
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
        self._default_weights = np.copy(weights)

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
        self._alpha = alpha
        self._delta_init = 1.0
        self._Delta = 1.0
        self._Delta_max = np.inf
        self._gamma = 0.0
        self._s = np.zeros(self._param_sum)
        self._delta = self._delta_init / self._Delta
        self._eps = self._delta

        self._g = 0
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        self.tsr_weights = torch.tensor(
            self._weights, dtype=torch.float64, requires_grad=True
        )
        self._weights_abs_sum = np.abs(self._weights).sum()
        self.lr_w = self._popsize**1.5 / 200
        self._n_epochs = n_epochs

        self._beta_q = 0.05

        self.tsr_s = torch.zeros(self._param_sum, dtype=torch.float64)
        self.tsr_gamma = torch.tensor(0.0, dtype=torch.float64)

        self.hat_SNRq = 0.0
        self.hat_signal = 0.0
        self.tsr_hat_signal = torch.tensor(0.0, dtype=torch.float64)

        self.ngrad_norm = 0.0

        self.archive = []
        self.archive_accum = [(self._s, self._gamma)]

        self._delta_archive = None

        self._update_interval = max(1, (self._n_ca**1.3) // self._popsize)

        self._t = 0

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
        cum_q = self._q.cumsum(axis=1)
        c = (cum_q - self._q <= rand_q) & (rand_q < cum_q)
        return c

    def tell(
        self, solutions: list[tuple[tuple[np.ndarray, np.ndarray], float]]
    ) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."

        self._g += 1
        self._t += 1
        solutions.sort(key=lambda s: s[1], reverse=True)

        self.avg_fvalue = np.mean([s[1] for s in solutions])

        self.old_delta = self._delta

        # Update of categorical distribution
        c = np.array([s[0] for s in solutions])
        ngrad = (self._weights[:, np.newaxis, np.newaxis] * (c - self._q)).sum(axis=0)
        self.ngrad_norm = np.linalg.norm(ngrad)

        archive_q = []
        for cc in c:
            archive_q.append(cc - self._q)
        self.archive.append(archive_q)

        sl = []
        for i, K in enumerate(self._K):
            q_i = self._q[i, : K - 1]
            q_i_K = self._q[i, K - 1]
            s_i = 1.0 / np.sqrt(q_i) * ngrad[i, : K - 1]
            s_i += np.sqrt(q_i) * ngrad[i, : K - 1].sum() / (q_i_K + np.sqrt(q_i_K))
            sl += list(s_i)
        ngrad_sqF = np.array(sl)

        pnorm = np.sqrt(np.dot(ngrad_sqF, ngrad_sqF)) + 1e-30
        self._eps = self._delta / pnorm
        self._q += self._eps * ngrad

        # Update of ASNG
        self._delta = self._delta_init / self._Delta

        if isinstance(self._delta_archive, list):
            self._delta_archive.append(self._delta)
        beta = self._delta / (self._param_sum**0.5)
        self.beta = beta

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
        self.delta_change_ratio = (self._delta - self.old_delta) / self.old_delta
        self.hat_SNRq = (np.linalg.norm(self._s)) ** 2 / self._gamma
        self.hat_signal = (np.linalg.norm(self._s)) ** 2 - self._gamma
        self.delta_exp = np.exp(
            beta * (self._gamma - np.dot(self._s, self._s) / self._alpha)
        )

        # Adapt weights
        if self._g % self._update_interval == 0:
            self.tsr_weights = torch.tensor(
                self._weights, dtype=torch.float64, requires_grad=True
            )
            optimizer = torch.optim.SGD([self.tsr_weights], lr=self.lr_w)
            scheduler = ExponentialLR(optimizer, gamma=0.9)
            tsr_beta = torch.tensor(beta, dtype=torch.float64)
            for _ in range(self._n_epochs):
                tsr_s = torch.tensor(self.archive_accum[-1][0], dtype=torch.float64)
                tsr_gamma = torch.tensor(self.archive_accum[-1][1], dtype=torch.float64)
                for archive_q in self.archive:
                    cur_delta_q = torch.tensor(np.array(archive_q), dtype=torch.float64)
                    tsr_Delta_q = self.tsr_weights[0] * cur_delta_q[0]
                    for i in range(1, self._popsize):
                        tsr_Delta_q += self.tsr_weights[i] * cur_delta_q[i]
                    tsr_sl = []
                    for i, K in enumerate(self._K):
                        q_i = self._q[i, : K - 1]
                        q_i_K = self._q[i, K - 1]
                        tsr_q_i = torch.tensor(q_i, dtype=torch.float64)
                        tsr_q_i_K = torch.tensor(q_i_K, dtype=torch.float64)
                        tsr_s_i = 1.0 / torch.sqrt(tsr_q_i) * tsr_Delta_q[i, : K - 1]
                        tsr_s_i += (
                            torch.sqrt(tsr_q_i)
                            * torch.sum(tsr_Delta_q[i, : K - 1])
                            / (tsr_q_i_K + torch.sqrt(tsr_q_i_K))
                        )
                        tsr_sl.append(tsr_s_i)
                    tsr_ngrad_sqF = torch.cat(tsr_sl)
                    dot_tsr_ngrad_sqF = torch.dot(tsr_ngrad_sqF, tsr_ngrad_sqF)
                    if dot_tsr_ngrad_sqF < 1e-30:
                        continue
                    tsr_pnorm = torch.sqrt(dot_tsr_ngrad_sqF) + 1e-30

                    # calculate SNR
                    tsr_s = (1 - tsr_beta) * tsr_s + torch.sqrt(
                        tsr_beta * (2 - tsr_beta)
                    ) * tsr_ngrad_sqF / tsr_pnorm
                    tsr_gamma = (1 - tsr_beta) ** 2 * tsr_gamma + tsr_beta * (
                        2 - tsr_beta
                    )
                self.tsr_hat_signal = (torch.linalg.norm(tsr_s)) ** 2 - tsr_gamma
                loss = -self.tsr_hat_signal
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                with torch.no_grad():
                    sorted_w = torch.sort(self.tsr_weights, descending=True)[0]
                    scaled_w = sorted_w / np.abs(sorted_w).sum() * self._weights_abs_sum
                    self.tsr_weights.copy_(scaled_w)
                self._weights = self.tsr_weights.clone().detach().numpy()

            if isinstance(self._delta_archive, list):
                self.mean_delta_archive = np.prod(self._delta_archive) ** (
                    1 / len(self._delta_archive)
                )
                self._update_interval = max(5, math.floor(self.mean_delta_archive**2))
                self._delta_archive = []
            self._t = 0

            self.archive_accum.append(
                (tsr_s.clone().detach().numpy(), tsr_gamma.clone().detach().numpy())
            )
            self.archive = []

        self.entropy = -np.sum(
            self._q[:, 0] * np.log2(self._q[:, 0])
            + self._q[:, 1] * np.log2(self._q[:, 1])
        )
