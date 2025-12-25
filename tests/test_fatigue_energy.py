import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "source"))

from compute_energy import compute_energy
from material_properties import MaterialProperties
from pff_model import PFFModel


def test_fatigue_reduces_fracture_energy():
    device = "cpu"
    # Simple triangular domain points (x, y)
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True, device=device)

    # Linear displacement field to generate non-zero strain
    u = inp[:, 0]
    v = 0.5 * inp[:, 1]
    alpha = (0.1 + 0.05 * inp[:, 0] + 0.02 * inp[:, 1]).requires_grad_()
    hist_alpha = torch.zeros_like(alpha)

    matprop = MaterialProperties(
        mat_E=torch.tensor(1.0, device=device),
        mat_nu=torch.tensor(0.3, device=device),
        w1=torch.tensor(1.0, device=device),
        l0=torch.tensor(0.1, device=device),
    )
    pffmodel = PFFModel(alpha_T=0.2, p_fatigue=1.0, G_f0=1.0)

    area_elem = torch.ones(inp.shape[0], device=device)

    _, E_d_pristine, _, _ = compute_energy(
        inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn=None, hist_alpha_bar=torch.zeros_like(alpha)
    )
    _, E_d_fatigued, _, _ = compute_energy(
        inp, u, v, alpha, hist_alpha, matprop, pffmodel, area_elem, T_conn=None, hist_alpha_bar=torch.ones_like(alpha)
    )

    # Fatigue accumulation should degrade fracture toughness and lower damage energy
    assert torch.all(E_d_fatigued < E_d_pristine)
