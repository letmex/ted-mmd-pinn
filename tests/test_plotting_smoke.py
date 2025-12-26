from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "source"))

from plotting import img_plot  # noqa: E402
from material_properties import MaterialProperties  # noqa: E402
from pff_model import PFFModel  # noqa: E402


class _DummyFieldComponent:
    def __init__(self):
        self.lmbda = torch.tensor(0.1)

    def fieldCalculation(self, inp):
        x = inp[..., -2]
        y = inp[..., -1]
        u = x + 0.1 * y
        v = y - 0.2 * x
        alpha = 0.5 * (x + y)
        return u, v, alpha


def test_img_plot_uses_spatial_coordinates_with_step(monkeypatch, tmp_path):
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    figdir = {"png": tmp_path / "png", "pdf": tmp_path / "pdf"}
    figdir["png"].mkdir()
    figdir["pdf"].mkdir()

    captured_coords = []
    import matplotlib.tri as tri  # Imported here to pick up the Agg backend configuration

    original_triangulation = tri.Triangulation

    def _recording_triangulation(x, y, *args, **kwargs):
        captured_coords.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return original_triangulation(x, y, *args, **kwargs)

    monkeypatch.setattr(tri, "Triangulation", _recording_triangulation)

    img_plot(
        field_comp=_DummyFieldComponent(),
        pffmodel=PFFModel(),
        matprop=MaterialProperties(
            mat_E=torch.tensor(1.0),
            mat_nu=torch.tensor(0.3),
            w1=torch.tensor(1.0),
            l0=torch.tensor(0.1),
        ),
        inp=coords,
        T=None,
        area_elem=torch.ones(coords.shape[0]),
        figdir=figdir,
        step_idx=1,
        total_steps=3,
    )

    # Two triangulations are expected (nodal field plot and stress plot)
    assert captured_coords, "No triangulations were constructed"
    for x_vals, y_vals in captured_coords:
        assert np.allclose(x_vals, [0.0, 1.0, 0.0])
        assert np.allclose(y_vals, [0.0, 0.0, 1.0])
