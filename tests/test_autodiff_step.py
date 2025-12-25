from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "source"))

from compute_energy import field_grads
from utils import append_step_column


def test_autodiff_gradients_ignore_step_dimension():
    # Base spatial input (x, y)
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 0.25]],
        requires_grad=True,
    )

    # Reference gradients without a step dimension
    field_base = (coords[:, 0] * coords[:, 1]) ** 2
    grad_ref = torch.autograd.grad(field_base.sum(), coords, create_graph=False)[0]

    # Append a step column and ensure gradients are taken w.r.t. spatial columns only
    coords_step = coords.detach().clone().requires_grad_(True)
    inp_with_step = append_step_column(coords_step, step_idx=1, total_steps=5)
    field_with_step = (inp_with_step[:, -2] * inp_with_step[:, -1]) ** 2 + 0.3 * inp_with_step[:, 0]

    grad_x, grad_y = field_grads(inp_with_step, field_with_step, area_elem=torch.tensor(1.0), T=None)

    assert torch.allclose(grad_x, grad_ref[:, 0])
    assert torch.allclose(grad_y, grad_ref[:, 1])
