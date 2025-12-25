import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
import torch
import numpy as np
import copy
from pathlib import Path

from compute_energy import gradients, stress, compute_energy
from utils import parse_mesh, append_step_column



def plot_mesh(mesh_file, figdir):
    X, Y, T, _ = parse_mesh(filename = mesh_file, gradient_type = 'numerical')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.triplot(X, Y, T, color='black', linewidth=1, rasterized=True)
    ax.set_axis_off()
    plt.savefig(figdir["png"]/Path('mesh.png'), transparent=True, bbox_inches='tight', dpi=600)
    plt.savefig(figdir["pdf"]/Path('mesh.pdf'), transparent=True, bbox_inches='tight', dpi=600)


def plot_field(inp, field, T, figname, figdir, dpi=300):
    input_pt = copy.deepcopy(inp)
    input_pt = input_pt.detach().numpy()
    triang = T
    if T == None:
        triang = tri.Triangulation(input_pt[:, 0], input_pt[:, 1]).triangles
        figname = figname + '-at-gp'

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_aspect('equal')
    tpc0 = ax.tripcolor(input_pt[:, 0], input_pt[:, 1], triang, field, shading='gouraud', rasterized=True)
    cbar = fig.colorbar(tpc0, ax = ax)
    cbar.formatter.set_powerlimits((0, 0))
    ax.set_title(figname)
    plt.savefig(figdir["png"]/Path(str(figname)+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path(str(figname)+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)


def plot_energy(field_comp, disp, pffmodel, matprop, inp, T_conn, area_elem,
                trainedModel_path, figdir, total_steps=None):
    energy = np.zeros([1, 2])
    n_steps = total_steps if total_steps is not None else len(disp)

    j = 0
    while True:
        model = trainedModel_path / Path(f'trained_1NN_{j}.pt')
        if not model.is_file():      # 注意这里用实例方法 is_file()
            break

        # 载入第 j 步训练好的网络
        field_comp.net.load_state_dict(
            torch.load(model, map_location=torch.device('cpu'))
        )
        field_comp.lmbda = torch.tensor(disp[j])

        # 如果输入里还没有 step 这一列，就补上
        inp_step = inp
        if inp_step.shape[1] == 2:
            inp_step = append_step_column(inp_step, step_idx=j, total_steps=n_steps)

        if T_conn is None:
            inp_step.requires_grad = True

        # 计算当前步的场变量
        field_outputs = field_comp.fieldCalculation(inp_step)
        u, v, alpha = field_outputs[0], field_outputs[1], field_outputs[2]

        # 计算能量：新版 compute_energy 返回 4 个量，我们只用前 2 个
        E_el, E_d, _, _ = compute_energy(
            inp_step, u, v, alpha,
            alpha,          # 这里 hist_alpha 简单取当前 alpha，用于后处理
            matprop, pffmodel, area_elem, T_conn
        )

        E_el, E_d = E_el.detach().numpy(), E_d.detach().numpy()
        energy = np.append(energy, np.array([[E_el, E_d]]), axis=0)
        j += 1

    if j > 0:
        energy = np.delete(energy, 0, 0)

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(disp[0:j], energy[0:j, 0], '-', label=r'$\mathcal{E}^{el}_{\theta}$')
        ax.plot(disp[0:j], energy[0:j, 1], '-', label=r'$\mathcal{E}^{d}_{\theta}$')
        ax.set_xlim((disp[0], disp[j-1]))
        ax.set_ylim((np.min(energy), np.max(energy)*1.1))
        ax.set_xlabel(r'$U_p$')
        ax.set_ylabel(r'$\mathcal{E}$')
        ax.legend(loc=2)
        plt.savefig(figdir["png"]/Path('energy_Up.png'), transparent=True, bbox_inches='tight')
        plt.savefig(figdir["pdf"]/Path('energy_Up.pdf'), transparent=True, bbox_inches='tight')
    else:
        print(f"No trained network available in {trainedModel_path}")



def img_plot(field_comp, pffmodel, matprop, inp, T, area_elem, figdir, dpi=300, step_idx=None, total_steps=None):
    input_pts = inp
    if step_idx is not None and input_pts.shape[1] == 2:
        n_steps = total_steps if total_steps is not None else step_idx+1
        input_pts = append_step_column(input_pts, step_idx=step_idx, total_steps=n_steps)
    if T == None:
        input_pts.requires_grad = True
    field_outputs = field_comp.fieldCalculation(input_pts)
    u, v, alpha = field_outputs[0], field_outputs[1], field_outputs[2]
    strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y = gradients(input_pts, u, v, alpha, area_elem, T)

    if T == None:
        input_elem = input_pts
        alpha_elem = alpha
    else:    
        input_elem = (input_pts[T[:, 0], :] + input_pts[T[:, 1], :] + input_pts[T[:, 2], :])/3
        alpha_elem = (alpha[T[:, 0]] + alpha[T[:, 1]] + alpha[T[:, 2]])/3
    stress_11, stress_22, stress_12 = stress(strain_11, strain_22, strain_12, alpha_elem, matprop, pffmodel) 

    stress_1 = 0.5*(stress_11 + stress_22) + torch.sqrt((0.5*(stress_11 - stress_22))**2 + stress_12**2)
    stress_2 = 0.5*(stress_11 + stress_22) - torch.sqrt((0.5*(stress_11 - stress_22))**2 + stress_12**2)

    input_pt = copy.deepcopy(input_pts)
    input_el = copy.deepcopy(input_elem)
    input_pt, input_el = input_pt.detach().numpy(), input_el.detach().numpy()
    u, v, alpha = u.detach().numpy(), v.detach().numpy(), alpha.detach().numpy()
    strain_11, strain_22, strain_12 = strain_11.detach().numpy(), strain_22.detach().numpy(), strain_12.detach().numpy()
    stress_11, stress_22, stress_12 = stress_11.detach().numpy(), stress_22.detach().numpy(), stress_12.detach().numpy()
    stress_1, stress_2 = stress_1.detach().numpy(), stress_2.detach().numpy()
    disp = field_comp.lmbda.item()

    if T == None:
        x = input_pt[:, -2]
        y = input_pt[:, -1]
        T = tri.Triangulation(x, y).triangles

    fig, ax = plt.subplots(figsize=(9.5, 2), ncols=3)
    ax[0].set_aspect('equal')
    tpc0 = ax[0].tripcolor(input_pt[:, 0], input_pt[:, 1], T, u, shading='gouraud', rasterized=True)
    cbar0 = fig.colorbar(tpc0, ax = ax[0])
    cbar0.formatter.set_powerlimits((0, 0))
    ax[0].set_title(r"$u_{\theta}$")

    ax[1].set_aspect('equal')
    tpc1 = ax[1].tripcolor(input_pt[:, 0], input_pt[:, 1], T, v, shading='gouraud', rasterized=True)
    cbar1 = fig.colorbar(tpc1, ax = ax[1])
    cbar1.formatter.set_powerlimits((0, 0))
    ax[1].set_title(r"$v_{\theta}$")

    ax[2].set_aspect('equal')
    tpc2 = ax[2].tripcolor(input_pt[:, 0], input_pt[:, 1], T, alpha, shading='gouraud', rasterized=True)
    cbar2 = fig.colorbar(tpc2, ax = ax[2])
    cbar2.formatter.set_powerlimits((0, 0))
    ax[2].set_title(r"$\alpha_{\theta}$")

    plt.savefig(figdir["png"]/Path('field_Up_'+str(int(disp*1000)) + 'by1000'+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path('field_Up_'+str(int(disp*1000)) + 'by1000'+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)


    # Stress plot
    x = input_el[:, 0]
    y = input_el[:, 1]
    triang = tri.Triangulation(x, y)
    triAnalyzer = tri.TriAnalyzer(triang)
    mask = triAnalyzer.get_flat_tri_mask(min_circle_ratio=0.1, rescale=False)
    triang.set_mask(mask)


    fig, ax = plt.subplots(figsize=(9.5, 2), ncols=3)
    ax[0].set_aspect('equal')
    tpc0 = ax[0].tripcolor(triang, stress_11, shading='gouraud', rasterized=True)
    cbar0 = fig.colorbar(tpc0, ax = ax[0])
    cbar0.formatter.set_powerlimits((0, 0))
    ax[0].set_title(r"$\sigma_{\theta_{11}}$")

    ax[1].set_aspect('equal')
    tpc1 = ax[1].tripcolor(triang, stress_22, shading='gouraud', rasterized=True)
    cbar1 = fig.colorbar(tpc1, ax = ax[1])
    cbar1.formatter.set_powerlimits((0, 0))
    ax[1].set_title(r"$\sigma_{\theta_{22}}$")

    ax[2].set_aspect('equal')
    tpc2 = ax[2].tripcolor(triang, stress_12, shading='gouraud', rasterized=True)
    cbar2 = fig.colorbar(tpc2, ax = ax[2])
    cbar2.formatter.set_powerlimits((0, 0))
    ax[2].set_title(r"$\sigma_{\theta_{12}}$")

    plt.savefig(figdir["png"]/Path('stress_Up_'+str(int(disp*1000)) + 'by1000'+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path('stress_Up_'+str(int(disp*1000)) + 'by1000'+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)


    # Principal stress plot
    fig, ax = plt.subplots(figsize=(6, 2), ncols=2)
    ax[0].set_aspect('equal')
    tpc0 = ax[0].tripcolor(triang, stress_1, shading='gouraud', rasterized=True)
    cbar0 = fig.colorbar(tpc0, ax = ax[0])
    cbar0.formatter.set_powerlimits((0, 0))
    ax[0].set_title(r"$\sigma_{\theta_1}$")

    ax[1].set_aspect('equal')
    tpc1 = ax[1].tripcolor(triang, stress_2, shading='gouraud', rasterized=True)
    cbar1 = fig.colorbar(tpc1, ax = ax[1])
    cbar1.formatter.set_powerlimits((0, 0))
    ax[1].set_title(r"$\sigma_{\theta_2}$")

    plt.savefig(figdir["png"]/Path('stress_p_Up_'+str(int(disp*1000)) + 'by1000'+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path('stress_p_Up_'+str(int(disp*1000)) + 'by1000'+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)
