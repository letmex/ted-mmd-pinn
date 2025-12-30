# Defines material properties
class MaterialProperties:
    def __init__(
        self,
        mat_E,
        mat_nu,
        w1,
        l0,
        ft=None,
        Gf0=None,
        b0=None,
        xi=None,
        c0=None,
        p=None,
        E_bar=None,
        lch_bar=None,
        thermal_props=None,
    ):
        self.mat_E = mat_E
        self.mat_nu = mat_nu
        self.w1 = w1
        self.l0 = l0
        self.ft = ft
        self.Gf0 = Gf0
        self.b0 = b0
        self.xi = xi
        self.c0 = c0
        self.p = p
        self.E_bar = E_bar
        self.lch_bar = lch_bar
        self.thermal_props = thermal_props or {}

        self.mat_lmbda = self.mat_E * self.mat_nu / (1 + self.mat_nu) / (1 - 2 * self.mat_nu)
        self.mat_mu = self.mat_E / (1 + self.mat_nu) / 2.0

    def __call__(self):
        return self.mat_lmbda, self.mat_mu, self.w1, self.l0
