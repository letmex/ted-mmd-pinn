import warnings
from typing import Optional

import torch

# Defines phase field fracture model
class PFFModel:
    def __init__(
        self,
        PFF_model='AT1',
        se_split='volumetric',
        tol_ir=5e-3,
        c0: float = 1.0,
        l0: float = 1.0,
        xi: float = 1.0,
        kappa_d: float = 1e-6,
        eta_S: float = 0.5,
        alpha_T: float = 0.05,
        p_fatigue: float = 1.0,
        G_f0: float = 1.0,
        k_T: float = 0.0,
        theta_0: float = 0.0,
        a1: float = 31.081,
        a2: float = -0.5,
        a3: float = 0.0,
        ck: float = 7.6394e-8,
        Psi_th: Optional[float] = None,
        m_theta: float = 0.8,
        alpha_max: float = 1.0,
        Y0: float = 1.0,
        Y_T: Optional[float] = None,
        Y_S: Optional[float] = None,
        H: Optional[float] = None,
    ):
        self.PFF_model = PFF_model
        self.se_split = se_split
        self.tol_ir = tol_ir
        self.c0 = c0
        self.l0 = l0
        self.xi = xi
        self.kappa_d = kappa_d
        self.eta_S = eta_S
        self.alpha_T = alpha_T
        self.p_fatigue = p_fatigue
        self.G_f0 = G_f0
        self.k_T = k_T
        self.theta_0 = theta_0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.ck = ck
        self.Y0 = Y0
        self.Psi_th = 0.3 * self.Y0 if Psi_th is None else Psi_th
        self.m_theta = m_theta
        self.alpha_max = alpha_max
        self.Y_T = self.Y0 + self.Psi_th if Y_T is None else Y_T
        self.Y_S = self.Y_T + self.Psi_th * self.m_theta if Y_S is None else Y_S
        self.H = self.Y0 if H is None else H
        
        if self.se_split != 'volumetric':
            warnings.warn(
                'Prescribed strain energy split is not volumetric. '
                'No strain energy split will be applied.'
            )
        
        if self.PFF_model not in ['AT1', 'AT2', 'TEF']:
            raise ValueError('PFF_model must be AT1, AT2, or TEF')

    # degradation function for Young's modulus and its derivative w.r.t. \alpha: g(\alpha) and g'(\alpha)
    def Edegrade(self, alpha):
        return self.degrade_Y_modulus(alpha)

    # damage function and its derivative w.r.t. \alpha: w(\alpha) and w'(\alpha) and c_w
    def damageFun(self, alpha):
        return self.crack_function(alpha)

    # crack geometric function and its derivative w.r.t. damage variable
    def crack_function(self, d):
        if self.PFF_model == 'AT1':
            alpha = d
            dalpha_dd = torch.ones_like(d)
            c_w = 8.0 / 3.0
        elif self.PFF_model == 'AT2':
            alpha = d**2
            dalpha_dd = 2 * d
            c_w = 2.0
        elif self.PFF_model == 'TEF':
            alpha, dalpha_dd, c_w = self.tef_damage_function(d)

        return alpha, dalpha_dd, c_w

    # degradation of Young's modulus and its derivative for effective stress and equivalent energy weights
    def degrade_Y_modulus(self, d):
        if self.PFF_model == 'TEF':
            g_d, dg_dd = self.tef_g_d(d)
        else:
            g_d = (1 - d)**2 + self.kappa_d
            dg_dd = 2 * (d - 1)
        return g_d, dg_dd

    # surface energy density and its derivatives
    def surface_energy(self, d, grad_d):
        if isinstance(grad_d, (list, tuple)):
            grad_normal = grad_d[0]
            grad_shear = grad_d[1] if len(grad_d) > 1 else torch.zeros_like(grad_normal)
            stack_output = False
        else:
            grad_normal = grad_d[..., 0]
            grad_shear = grad_d[..., 1] if grad_d.shape[-1] > 1 else torch.zeros_like(grad_normal)
            stack_output = True

        grad_normal_sq = grad_normal**2
        grad_shear_sq = grad_shear**2
        grad_term = (1 - self.xi) * grad_normal_sq + self.xi * (grad_normal_sq + self.eta_S * grad_shear_sq)

        gamma = self.c0 * ((self.kappa_d + d**2) / (2 * self.l0) + 0.5 * self.l0 * grad_term)
        dgamma_dd = self.c0 * d / self.l0
        dgamma_dgrad_normal = self.c0 * self.l0 * grad_normal
        dgamma_dgrad_shear = self.c0 * self.l0 * self.xi * self.eta_S * grad_shear

        if stack_output:
            dgamma_dgrad = torch.stack((dgamma_dgrad_normal, dgamma_dgrad_shear), dim=-1)
        else:
            dgamma_dgrad = (dgamma_dgrad_normal, dgamma_dgrad_shear)

        return gamma, dgamma_dd, dgamma_dgrad

    # fatigue degradation of fracture toughness
    def fatigue_degrade(self, alpha_bar, Y_bar=None):
        if self.PFF_model == 'TEF':
            return self.tef_fatigue(alpha_bar, Y_bar)
        alpha_bar_clamped = torch.clamp(alpha_bar, min=0.0)
        alpha_T = torch.as_tensor(self.alpha_T, device=alpha_bar.device, dtype=alpha_bar.dtype)
        one_minus_alpha_T = torch.clamp(1 - alpha_T, min=torch.tensor(1e-8, device=alpha_bar.device, dtype=alpha_bar.dtype))
        fatigue_exponent = ((alpha_bar_clamped - alpha_T) / one_minus_alpha_T).clamp(min=0.0)
        decay = torch.exp(-fatigue_exponent**self.p_fatigue)
        fatigue_factor = torch.where(
            alpha_bar_clamped < alpha_T,
            torch.ones_like(alpha_bar_clamped),
            decay
        )
        return self.G_f0 * fatigue_factor

    # optional thermal acceleration factor (based on temperature theta)
    def temperature_boost(self, theta):
        if self.k_T == 0.0:
            return torch.ones_like(theta)
        return torch.exp(self.k_T * (theta - self.theta_0))

    # Irreversibility penalty
    def irrPenalty(self):
        if self.PFF_model == 'AT1':
            return 27 / 64 / self.tol_ir**2
        elif self.PFF_model in ['AT2', 'TEF']:
            return 1.0 / self.tol_ir**2 - 1.0

    def tef_damage_function(self, d):
        omega, domega_dd, _, _ = self.compute_omega(d)
        return omega, domega_dd, 1.0

    def tef_g_d(self, d):
        one = torch.as_tensor(1.0, device=d.device, dtype=d.dtype)
        g_d = (one - d) ** 2 + torch.as_tensor(1e-4, device=d.device, dtype=d.dtype)
        dg_dd = -2 * (one - d)
        return g_d, dg_dd

    def compute_omega(self, d):
        a1 = torch.as_tensor(self.a1, device=d.device, dtype=d.dtype)
        a2 = torch.as_tensor(self.a2, device=d.device, dtype=d.dtype)
        a3 = torch.as_tensor(self.a3, device=d.device, dtype=d.dtype)
        fai = torch.exp(a1 * (1.0 - d))
        dfai_dd = -a1 * fai
        gai = a1 * d + (1 + a2 * d + a3 * d * d)
        dgai_dd = a1 + a2 + 2 * a3 * d
        denom = fai + gai
        omega = fai / denom
        domega_dd = (dfai_dd * denom - fai * (dfai_dd + dgai_dd)) / (denom**2)
        return omega, domega_dd, fai, gai

    def tef_fatigue(self, alpha_bar, Y_bar=None):
        alpha_bar_clamped = torch.clamp(alpha_bar, min=0.0, max=self.alpha_max)
        ph = self.compute_ph(alpha_bar_clamped, Y_bar)
        ff = torch.exp(-torch.as_tensor(self.ck, device=alpha_bar.device, dtype=alpha_bar.dtype) * ph**self.p_fatigue)
        Gf = self.G_f0 * ff
        return Gf, {"ph": ph, "ff": ff}

    def compute_ph(self, alpha_bar, Y_bar=None):
        if Y_bar is None:
            Y_bar = torch.zeros_like(alpha_bar)
        Y_T = torch.as_tensor(self.Y_T, device=alpha_bar.device, dtype=alpha_bar.dtype)
        Y_S = torch.as_tensor(self.Y_S, device=alpha_bar.device, dtype=alpha_bar.dtype)
        denom = torch.clamp(Y_S - Y_T, min=torch.finfo(alpha_bar.dtype).eps)
        base = torch.clamp((Y_bar - Y_T) / denom, min=0.0)
        active = torch.where(alpha_bar < self.alpha_T, torch.zeros_like(base), base)
        return active
