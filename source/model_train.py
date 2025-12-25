import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path

from input_data_from_mesh import prep_input_data
from fit import fit, fit_with_early_stopping
from optim import *
from plotting import plot_field
from compute_energy import compute_energy

def train(field_comp, disp, pffmodel, matprop, crack_dict, numr_dict, optimizer_dict, training_dict, coarse_mesh_file, fine_mesh_file, device, trainedModel_path, intermediateModel_path, writer):
    '''
    Neural network training: pretraining with a coarser mesh in the first stage before the main training proceeds.
    
    Input is prepared from the .msh file.

    Network training to learn the solution of the BVP in step wise loading.
    Trained network from the previous load step is used for learning the solution
    in the current load step.

    Trained models and loss data are saved in the trainedModel_path directory.
    '''
    
    ## #############################################################################
    # Initial training #############################################################
    # Prepare initial input data
    inp, T_conn, area_T, hist_alpha = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file=coarse_mesh_file, device=device)
    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)
    field_comp.lmbda = torch.tensor(disp[0]).to(device)

    loss_data = list()
    start = time.time()

    n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "LBFGS")
    loss_data1 = fit(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                     optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer, 
                     intermediateModel_path=None, writer=writer, training_dict=training_dict)
    loss_data = loss_data + loss_data1

    n_epochs = optimizer_dict["n_epochs_RPROP"]
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "RPROP")
    loss_data2 = fit_with_early_stopping(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                                         optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer, min_delta=optimizer_dict["optim_rel_tol_pretrain"], 
                                         intermediateModel_path=None, writer=writer, training_dict=training_dict)
    loss_data = loss_data + loss_data2

    end = time.time()
    print(f"Execution time: {(end-start)/60:.03f}minutes")

    torch.save(field_comp.net.state_dict(), trainedModel_path/Path('trained_1NN_initTraining.pt'))
    with open(trainedModel_path/Path('trainLoss_1NN_initTraining.npy'), 'wb') as file:
        np.save(file, np.asarray(loss_data))

    ## #############################################################################


    ## #############################################################################
    # Main training ################################################################

    # Prepare input data
    inp, T_conn, area_T, hist_alpha = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file=fine_mesh_file, device=device)
    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)

    # solve BVP by step wise loading.
    for j, disp_i in enumerate(disp):
        print(f'idx: {j}; displacement: {disp_i}')
        loss_data = list()

        target_disp = disp_i
        attempt = 0
        prev_state = copy.deepcopy(field_comp.net.state_dict())

        while True:
            start = time.time()
            field_comp.net.load_state_dict(prev_state)
            field_comp.lmbda = torch.tensor(target_disp).to(device)
        
            loss_data_attempt = list()

            if training_dict.get("adaptive_loading", False):
                energy_before = _compute_total_energy(field_comp, inp, T_conn, area_T, hist_alpha, matprop, pffmodel)
            else:
                energy_before = None

            if j == 0 or optimizer_dict["n_epochs_LBFGS"] > 0:
                n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
                NNparams = field_comp.net.parameters()
                optimizer = get_optimizer(NNparams, "LBFGS")
                loss_data1 = fit(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                                 optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer,
                                 intermediateModel_path=None, writer=writer, training_dict=training_dict)
                loss_data_attempt = loss_data_attempt + loss_data1

            if optimizer_dict["n_epochs_RPROP"] > 0:
                n_epochs = optimizer_dict["n_epochs_RPROP"]
                NNparams = field_comp.net.parameters()
                optimizer = get_optimizer(NNparams, "RPROP")
                loss_data2 = fit_with_early_stopping(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                                                     optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer, min_delta=optimizer_dict["optim_rel_tol"],
                                                     intermediateModel_path=intermediateModel_path, writer=writer, training_dict=training_dict)
                loss_data_attempt = loss_data_attempt + loss_data2

            energy_after = _compute_total_energy(field_comp, inp, T_conn, area_T, hist_alpha, matprop, pffmodel)

            if training_dict.get("adaptive_loading", False) and energy_before is not None:
                if energy_after > energy_before + training_dict.get("energy_tol", 0.0) and attempt < training_dict.get("max_cycles", 1):
                    attempt += 1
                    target_disp = _reduced_disp(disp, j, target_disp, training_dict)
                    print(f"Energy increased from {energy_before} to {energy_after}. Reducing load increment to {target_disp} (attempt {attempt}).")
                    continue

            loss_data = loss_data_attempt
            break

        end = time.time()
        print(f"Execution time: {(end-start)/60:.03f}minutes")

        hist_alpha = field_comp.update_hist_alpha(inp)

        torch.save(field_comp.net.state_dict(), trainedModel_path/Path('trained_1NN_' + str(j) + '.pt'))
        with open(trainedModel_path/Path('trainLoss_1NN_' + str(j) + '.npy'), 'wb') as file:
            np.save(file, np.asarray(loss_data))


def _compute_total_energy(field_comp, inp, T_conn, area_T, hist_alpha, matprop, pffmodel):
    inp_energy = inp.clone().detach()
    if T_conn == None:
        inp_energy.requires_grad = True
        u, v, alpha = field_comp.fieldCalculation(inp_energy)
        energy_el, energy_d, energy_hist = compute_energy(inp_energy, u, v, alpha, hist_alpha, matprop, pffmodel, area_T, T_conn)
    else:
        with torch.no_grad():
            u, v, alpha = field_comp.fieldCalculation(inp_energy)
            energy_el, energy_d, energy_hist = compute_energy(inp_energy, u, v, alpha, hist_alpha, matprop, pffmodel, area_T, T_conn)
    return (energy_el + energy_d + energy_hist).item()


def _reduced_disp(disp, idx, current_disp, training_dict):
    step_factor = training_dict.get("fatigue_cycle_step", 0.5)
    if idx == 0:
        return current_disp * step_factor
    previous_disp = disp[idx-1]
    return previous_disp + (current_disp - previous_disp) * step_factor
