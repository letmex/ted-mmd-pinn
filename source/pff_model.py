import warnings
import torch

# Defines phase field fracture model
class PFFModel:
    def __init__(
        self,
        PFF_model = 'AT1',
        se_split = 'volumetric',
        tol_ir = 5e-3,
        c0: float = 1.0,
        l0: float = 1.0,
        xi: float = 1.0,
        kappa_d: float = 1e-6,
        eta_S: float = 1.0,
        alpha_T: float = 1.0,
        p_fatigue: float = 1.0,
        G_f0: float = 1.0,
        k_T: float = 0.0,
        theta_0: float = 0.0,
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
        
        if self.se_split != 'volumetric':
            warnings.warn('Prescribed strain energy split is not volumetric. No strain energy split will be applied.')
        
        if self.PFF_model not in ['AT1', 'AT2']:
            raise ValueError('PFF_model must be AT1 or AT2')

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
            c_w = 8.0/3.0
        elif self.PFF_model == 'AT2':
            alpha = d**2
            dalpha_dd = 2*d
            c_w = 2.0

        return alpha, dalpha_dd, c_w

    # degradation of Young's modulus and its derivative for effective stress and equivalent energy weights
    def degrade_Y_modulus(self, d):
        g_d = (1 - d)**2 + self.kappa_d
        dg_dd = 2*(d - 1)

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
        grad_term = (1 - self.xi)*grad_normal_sq + self.xi*(grad_normal_sq + self.eta_S*grad_shear_sq)

        gamma = self.c0*((self.kappa_d + d**2)/(2*self.l0) + 0.5*self.l0*grad_term)
        dgamma_dd = self.c0*d/self.l0
        dgamma_dgrad_normal = self.c0*self.l0*grad_normal
        dgamma_dgrad_shear = self.c0*self.l0*self.xi*self.eta_S*grad_shear

        if stack_output:
            dgamma_dgrad = torch.stack((dgamma_dgrad_normal, dgamma_dgrad_shear), dim=-1)
        else:
            dgamma_dgrad = (dgamma_dgrad_normal, dgamma_dgrad_shear)

        return gamma, dgamma_dd, dgamma_dgrad

    # fatigue degradation of fracture toughness
    def fatigue_degrade(self, alpha_bar):
        alpha_bar_clamped = torch.clamp(alpha_bar, min=0.0)
        fatigue_exponent = ((alpha_bar_clamped - self.alpha_T)/torch.clamp(1 - self.alpha_T, min=1e-8)).clamp(min=0.0)
        decay = torch.exp(-fatigue_exponent**self.p_fatigue)
        fatigue_factor = torch.where(alpha_bar_clamped < self.alpha_T, torch.ones_like(alpha_bar_clamped), decay)

        return self.G_f0*fatigue_factor

    # optional thermal acceleration factor
    def temperature_boost(self, theta):
        if self.k_T == 0.0:
            return torch.ones_like(theta)

        return torch.exp(self.k_T*(theta - self.theta_0))
    
    # Irreversibility penalty
    def irrPenalty(self):
        if self.PFF_model == 'AT1':
            return 27/64/self.tol_ir**2
        elif self.PFF_model == 'AT2':
            return 1.0/self.tol_ir**2-1.0
