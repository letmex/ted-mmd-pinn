
import torch


# Defines material properties
class MaterialProperties:
    def __init__(self, mat_E, mat_nu, w1, l0, ft=None, Gf0=None, b0=None, xi=None, c0=None, p=None, E_bar=None, lch_bar=None):
        self.mat_E = mat_E
        self.mat_nu = mat_nu
        self.w1 = w1
        self.l0 = l0
        self.mat_lmbda = self.mat_E*self.mat_nu/(1+self.mat_nu)/(1-2*self.mat_nu)
        self.mat_mu = self.mat_E/(1+self.mat_nu)/2.0

        self.ft = self._maybe_tensor(ft)
        self.Gf0 = self._maybe_tensor(Gf0)
        self.b0 = self._maybe_tensor(b0)
        self.xi = self._maybe_tensor(xi)
        self.c0 = self._maybe_tensor(c0)
        self.p = self._maybe_tensor(p)
        self.E_bar = self._maybe_tensor(E_bar)
        self.lch_bar = self._maybe_tensor(lch_bar)

    def _maybe_tensor(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(value, device=self.mat_E.device, dtype=self.mat_E.dtype)

    def __call__(self):
        return self.mat_lmbda, self.mat_mu, self.w1, self.l0
    
