
import torch

# Defines material properties
class MaterialProperties:
    def __init__(self, mat_E, mat_nu, w1, l0, thermal_props=None):
        self.mat_E = mat_E
        self.mat_nu = mat_nu
        self.w1 = w1
        self.l0 = l0
        self.mat_lmbda = self.mat_E*self.mat_nu/(1+self.mat_nu)/(1-2*self.mat_nu)
        self.mat_mu = self.mat_E/(1+self.mat_nu)/2.0

        self.thermal_props = self._init_thermal_props(thermal_props, mat_E.device, mat_E.dtype)

    def _init_thermal_props(self, thermal_props, device, dtype):
        if thermal_props is None:
            return None

        def _to_tensor(val, default=0.0):
            if val is None:
                val = default
            return torch.as_tensor(val, device=device, dtype=dtype)

        return {
            "alpha": _to_tensor(thermal_props.get("alpha") if isinstance(thermal_props, dict) else None),
            "rho": _to_tensor(thermal_props.get("rho") if isinstance(thermal_props, dict) else None),
            "k0": _to_tensor(thermal_props.get("k0") if isinstance(thermal_props, dict) else None),
            "c": _to_tensor(thermal_props.get("c") if isinstance(thermal_props, dict) else None),
        }

    def __call__(self):
        return self.mat_lmbda, self.mat_mu, self.w1, self.l0
    
