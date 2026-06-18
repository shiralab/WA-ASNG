import numpy as np


class OneMax:
    def __init__(self, cat_dim: int, cat_num: np.ndarray):
        self.D = cat_dim
        self.cat_num = cat_num
        self.best_fvalue = self.D

    def _evaluate(self, c):
        return sum(c[:, 1])


class LeadingOnes:
    def __init__(self, cat_dim: int, cat_num: np.ndarray):
        self.D = cat_dim
        self.cat_num = cat_num
        self.best_fvalue = self.D

    def _evaluate(self, c):
        return c[:, 1].argmin() + c[:, 1].prod() * self.D


class BinVal:
    def __init__(self, cat_dim: int, cat_num: np.ndarray):
        self.D = cat_dim
        self.cat_num = cat_num
        self.K = int(cat_num[0])
        self.best_fvalue = self.K**self.D - 1

    def _evaluate(self, c):
        k_indices = np.arange(self.K)
        k_coeff = k_indices

        d_indices = np.arange(1, self.D + 1).reshape(self.D, 1)
        k_exp = self.K ** (self.D - d_indices.astype(object))

        weights = k_coeff.astype(object) * k_exp
        kval = 0
        for i in range(self.D):
            k = np.where(c[i] == 1)[0][0]
            kval += weights[i, k]
        return kval


class NoisyOneMax(OneMax):
    def __init__(self, cat_dim: int, cat_num: np.ndarray, noisevar: float, seed: int):
        super().__init__(cat_dim, cat_num)
        self.noisevar = noisevar
        self._rng = np.random.RandomState(seed)

    def _evaluate(self, c):
        true_fval = super()._evaluate(c)
        if self.noisevar == "cauchy":
            noisy_fval = true_fval + self._rng.standard_cauchy()
        else:
            noisy_fval = true_fval + self._rng.normal(0.0, np.sqrt(self.noisevar))
        return true_fval, noisy_fval


class NoisyLeadingOnes(LeadingOnes):
    def __init__(self, cat_dim: int, cat_num: np.ndarray, noisevar: float, seed: int):
        super().__init__(cat_dim, cat_num)
        self.noisevar = noisevar
        self._rng = np.random.RandomState(seed)

    def _evaluate(self, c):
        true_fval = super()._evaluate(c)
        if self.noisevar == "cauchy":
            noisy_fval = true_fval + np.random.standard_cauchy()
        else:
            noisy_fval = true_fval + self._rng.normal(0.0, np.sqrt(self.noisevar))
        return true_fval, noisy_fval


class NoisyBinVal(BinVal):
    def __init__(self, cat_dim: int, cat_num: np.ndarray, noisevar: float, seed: int):
        super().__init__(cat_dim, cat_num)
        self.noisevar = noisevar
        self._rng = np.random.RandomState(seed)

    def _evaluate(self, c):
        true_fval = super()._evaluate(c)
        if self.noisevar == "cauchy":
            noisy_fval = true_fval + np.random.standard_cauchy()
        else:
            noisy_fval = true_fval + self._rng.normal(0.0, np.sqrt(self.noisevar))
        return true_fval, noisy_fval
