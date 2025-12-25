import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path

from input_data_from_mesh import prep_input_data
from compute_energy import compute_energy
from fit import fit, fit_with_early_stopping
from optim import *
from plotting import plot_field
from utils import append_step_column


def train(field_comp, load_schedule, pffmodel, matprop, crack_dict, numr_dict,
          optimizer_dict, training_dict, coarse_mesh_file, fine_mesh_file,
          device, trainedModel_path, intermediateModel_path, writer):
    '''
    Neural network training: pretraining with a coarser mesh in the first stage
    before the main training proceeds.

    Input is prepared from the .msh file.

    Network training to learn the solution of the BVP in step wise loading.
    Trained network from the previous load step is used for learning the solution
    in the current load step.

    Trained models and loss data are saved in the trainedModel_path directory.
    '''
    disp = load_schedule["displacement"]
    temperature_steps = load_schedule.get("temperature", None)
    cycle_steps = load_schedule.get("cycles", None)
    total_steps = len(disp)
    assert temperature_steps is None or len(temperature_steps) == total_steps, \
        "temperature schedule length must match displacement schedule"
    assert cycle_steps is None or len(cycle_steps) == total_steps, \
        "cycle schedule length must match displacement schedule"

    # ###########################################################################
    # Initial training ##########################################################
    # 使用粗网格做一次预训练

    # base_inp: 只包含空间坐标；后面用 append_step_column 加上 step 维度
    base_inp, T_conn, area_T, hist_alpha = prep_input_data(
        matprop, pffmodel, crack_dict, numr_dict,
        mesh_file=coarse_mesh_file, device=device
    )
    # 疲劳辅助量的历史最大值 H，初始为 0
    hist_Y_max_over_H = torch.zeros_like(hist_alpha)

    # 第 0 步的输入：在 base_inp 上加一个 step 列
    inp = append_step_column(base_inp, step_idx=0, total_steps=total_steps)

    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(
        torch.utils.data.TensorDataset(inp, outp),
        batch_size=inp.shape[0],
        shuffle=False
    )

    field_comp.lmbda = torch.tensor(disp[0]).to(device)
    field_comp.temperature = torch.tensor(
        temperature_steps[0] if temperature_steps is not None else 0.0
    ).to(device)
    field_comp.cycle = torch.tensor(
        cycle_steps[0] if cycle_steps is not None else 0.0
    ).to(device)

    loss_data = list()
    start = time.time()

    # 先用 LBFGS 预训练
    n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "LBFGS")
    loss_data1 = fit(
        field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
        optimizer_dict["weight_decay"], num_epochs=n_epochs,
        optimizer=optimizer,
        hist_Y_max_over_H=hist_Y_max_over_H,
        intermediateModel_path=None,
        writer=writer,
        training_dict=training_dict
    )
    loss_data = loss_data + loss_data1

    # 再用 RPROP + early stopping
    n_epochs = optimizer_dict["n_epochs_RPROP"]
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "RPROP")
    loss_data2 = fit_with_early_stopping(
        field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
        optimizer_dict["weight_decay"], num_epochs=n_epochs,
        optimizer=optimizer,
        min_delta=optimizer_dict["optim_rel_tol_pretrain"],
        hist_Y_max_over_H=hist_Y_max_over_H,
        intermediateModel_path=None,
        writer=writer,
        training_dict=training_dict
    )
    loss_data = loss_data + loss_data2

    end = time.time()
    print(f"Execution time: {(end-start)/60:.03f} minutes")

    torch.save(field_comp.net.state_dict(),
               trainedModel_path / Path('trained_1NN_initTraining.pt'))
    with open(trainedModel_path / Path('trainLoss_1NN_initTraining.npy'), 'wb') as file:
        np.save(file, np.asarray(loss_data))

    # ###########################################################################
    # Main training #############################################################

    # 使用细网格进行主训练
    base_inp, T_conn, area_T, hist_alpha = prep_input_data(
        matprop, pffmodel, crack_dict, numr_dict,
        mesh_file=fine_mesh_file, device=device
    )
    # 疲劳历史变量
    hist_alpha_bar = torch.zeros_like(hist_alpha)
    hist_Y_max_over_H = torch.zeros_like(hist_alpha)

    outp = torch.zeros(base_inp.shape[0], 1).to(device)

    # 按位移/温度/循环次数逐步加载
    for j, disp_i in enumerate(disp):
        # 当前步的输入：在 base_inp 上加 step 列
        inp = append_step_column(base_inp, step_idx=j, total_steps=total_steps)
        training_set = DataLoader(
            torch.utils.data.TensorDataset(inp, outp),
            batch_size=inp.shape[0],
            shuffle=False
        )

        field_comp.lmbda = torch.tensor(disp_i).to(device)
        field_comp.temperature = torch.tensor(
            temperature_steps[j] if temperature_steps is not None else 0.0
        ).to(device)
        field_comp.cycle = torch.tensor(
            cycle_steps[j] if cycle_steps is not None else 0.0
        ).to(device)
        print(f'idx: {j}; displacement: {field_comp.lmbda}')
        loss_data = list()

        start = time.time()

        # 每步先跑 LBFGS（如果设置了）
        if j == 0 or optimizer_dict["n_epochs_LBFGS"] > 0:
            n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
            NNparams = field_comp.net.parameters()
            optimizer = get_optimizer(NNparams, "LBFGS")
            loss_data1 = fit(
                field_comp, training_set, T_conn, area_T, hist_alpha, matprop,
                pffmodel, optimizer_dict["weight_decay"], num_epochs=n_epochs,
                optimizer=optimizer,
                hist_Y_max_over_H=hist_Y_max_over_H,
                intermediateModel_path=None,
                writer=writer,
                training_dict=training_dict
            )
            loss_data = loss_data + loss_data1

        # 再跑 RPROP + early stopping（如果设置了）
        if optimizer_dict["n_epochs_RPROP"] > 0:
            n_epochs = optimizer_dict["n_epochs_RPROP"]
            NNparams = field_comp.net.parameters()
            optimizer = get_optimizer(NNparams, "RPROP")
            loss_data2 = fit_with_early_stopping(
                field_comp, training_set, T_conn, area_T, hist_alpha, matprop,
                pffmodel, optimizer_dict["weight_decay"], num_epochs=n_epochs,
                optimizer=optimizer,
                min_delta=optimizer_dict["optim_rel_tol"],
                hist_Y_max_over_H=hist_Y_max_over_H,
                intermediateModel_path=intermediateModel_path,
                writer=writer,
                training_dict=training_dict
            )
            loss_data = loss_data + loss_data2

        end = time.time()
        print(f"Execution time: {(end-start)/60:.03f} minutes")

        # ---------------- 疲劳累积与历史量更新 ----------------
        with torch.no_grad():
            # 当前步的场解
            field_outputs = field_comp.fieldCalculation(inp)
            u_curr, v_curr, alpha_curr = field_outputs[0], field_outputs[1], field_outputs[2]

            # 计算 TEF 能量及疲劳驱动力 Y_bar
            _, _, _, Y_bar = compute_energy(
                inp, u_curr, v_curr, alpha_curr,
                hist_alpha, matprop, pffmodel,
                area_T, T_conn, hist_Y_max_over_H
            )

            # 温度放大因子（可为常数或场）
            temp_boost = pffmodel.temperature_boost(inp)
            if not torch.is_tensor(temp_boost):
                temp_boost = torch.tensor(
                    temp_boost, device=inp.device, dtype=inp.dtype
                )
            temp_boost = temp_boost.view(-1) if temp_boost.ndim > 0 else temp_boost
            if temp_boost.numel() == 1:
                temp_boost = temp_boost.expand_as(Y_bar)

            # 疲劳速率 & 累积（截断到 1.0）
            alpha_rate = Y_bar * temp_boost
            hist_alpha_bar = torch.clamp(hist_alpha_bar + alpha_rate, max=1.0)

            # TEF 历史量 H 取最大值
            hist_Y_max_over_H = torch.maximum(hist_Y_max_over_H, Y_bar)

        # 更新 hist_alpha（兼容返回 tuple 的新版接口）
        hist_update = field_comp.update_hist_alpha(inp)
        if isinstance(hist_update, tuple):
            hist_alpha, aux_state = hist_update
            field_comp.hist_aux_state = aux_state
        else:
            hist_alpha = hist_update
            field_comp.hist_aux_state = {}

        # -----------------------------------------------------

        torch.save(field_comp.net.state_dict(),
                   trainedModel_path / Path(f'trained_1NN_{j}.pt'))
        with open(trainedModel_path / Path(f'trainLoss_1NN_{j}.npy'), 'wb') as file:
            np.save(file, np.asarray(loss_data))
