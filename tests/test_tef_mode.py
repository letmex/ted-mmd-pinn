import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "source"))

from compute_energy import compute_energy
from material_properties import MaterialProperties
from pff_model import PFFModel


def test_tef_material_functions_and_fatigue_scaling():
    pffmodel = PFFModel(
        PFF_model="TEF",
        a1=2.0,
        a2=0.5,
        a3=-0.25,
        ck=1.0,
        Y0=1.0,
        Y_T=0.2,
        Y_S=1.2,
        alpha_T=0.0,
        G_f0=2.0,
    )
    d = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

    g_d, _ = pffmodel.Edegrade(d)
    assert torch.allclose(g_d, (1 - d) ** 2 + 1e-4)

    omega, _, _, _ = pffmodel.compute_omega(d)
    assert omega[0] > omega[-1]

    alpha_bar = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    Y_bar = torch.tensor([0.0, 0.7, 1.0], dtype=torch.float64)
    Gf, aux = pffmodel.fatigue_degrade(alpha_bar, Y_bar=Y_bar)

    ph_expected = torch.clamp((Y_bar - 0.2) / (1.2 - 0.2), min=0.0)
    ff_expected = torch.exp(-ph_expected)
    assert torch.allclose(Gf, pffmodel.G_f0 * ff_expected)
    assert torch.allclose(aux["ph"], ph_expected)


def test_tef_fatigue_lowers_damage_energy():
    device = "cpu"
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True, device=device)
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
    pffmodel = PFFModel(
        PFF_model="TEF",
        a1=5.0,
        ck=2.0,
        Y0=1.0,
        Y_T=0.2,
        Y_S=1.2,
        alpha_T=0.05,
        G_f0=1.0,
    )

    area_elem = torch.ones(inp.shape[0], device=device)

    _, E_d_pristine, _, _ = compute_energy(
        inp,
        u,
        v,
        alpha,
        hist_alpha,
        matprop,
        pffmodel,
        area_elem,
        T_conn=None,
        hist_alpha_bar=torch.zeros_like(alpha),
    )
    _, E_d_fatigued, _, _ = compute_energy(
        inp,
        u,
        v,
        alpha,
        hist_alpha,
        matprop,
        pffmodel,
        area_elem,
        T_conn=None,
        hist_alpha_bar=torch.ones_like(alpha),
    )

    assert torch.all(E_d_fatigued < E_d_pristine)
