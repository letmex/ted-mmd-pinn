import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from compute_energy import compute_energy


class EarlyStopping:
    '''
    If the relative decrease in the loss is < min_delta for # of consecutive steps = tolerance,
    then the training is stopped.
    '''
    def __init__(self, tol_steps=10, min_delta=1e-3, device='cpu'):
        self.tol_steps = torch.tensor([tol_steps], dtype=torch.int, device=device)
        self.min_delta = torch.tensor([min_delta], dtype=torch.float, device=device)
        self.counter = torch.tensor([0], dtype=torch.int, device=device)
        self.early_stop = False        
        
    def __call__(self, train_loss, train_loss_prev):
        delta = torch.abs(train_loss - train_loss_prev) / (torch.abs(train_loss_prev) + np.finfo(float).eps)
        if delta > self.min_delta:
            self.counter = self.counter * 0
        else:
            self.counter += 1
            if self.counter >= self.tol_steps:  
                self.early_stop = True



def fit(field_comp, training_set_collocation, T_conn, area_T, hist_alpha, matprop, pffmodel,
        weight_decay, num_epochs, optimizer, hist_Y_max_over_H=None, hist_alpha_bar=None,
        intermediateModel_path=None, writer=None, training_dict={}):
    loss_data = list()
    step_counter = 0

    # Loop over epochs
    for epoch in range(num_epochs):
        loop = tqdm(training_set_collocation, miniters=25)
        # Loop over batches
        for j, (inp_train, outp_train) in enumerate(loop):

            loss_cache = {}

            def closure():
                optimizer.zero_grad()
                if T_conn is None:
                    inp_train.requires_grad = True

                # 兼容 fieldCalculation 返回多个量的写法
                field_outputs = field_comp.fieldCalculation(inp_train)
                u, v, alpha = field_outputs[0], field_outputs[1], field_outputs[2]

                # 新版 compute_energy，带 hist_Y_max_over_H，返回 4 个量
                loss_E_el, loss_E_d, loss_hist, _ = compute_energy(
                    inp_train, u, v, alpha,
                    hist_alpha, matprop, pffmodel,
                    area_T, T_conn, hist_Y_max_over_H, hist_alpha_bar
                )

                # Normalize the energy by total measure inside compute_energy to keep
                # elastic and damage contributions on comparable scales.
                loss_var = loss_E_el + loss_E_d + loss_hist

                # weight regularization
                loss_reg = 0.0
                if weight_decay != 0:
                    for name, param in field_comp.net.named_parameters():
                        if 'weight' in name:
                            loss_reg += torch.sum(param**2)

                loss = loss_var + weight_decay * loss_reg

                loss_cache["loss"] = loss.detach()
                loss_cache["loss_E"] = loss_var.detach()

                loss.backward()
                return loss

            loss = optimizer.step(closure=closure)
            step_counter += 1

            # Retrieve cached loss values from the last closure call
            loss_value = loss_cache.get("loss")
            loss_E_value = loss_cache.get("loss_E", None)
            if loss_value is None:
                loss_value = loss.detach() if torch.is_tensor(loss) else torch.tensor(loss)

            if writer is not None and loss_E_value is not None:
                writer.add_scalars(
                    'U_p_' + str(field_comp.lmbda.item()),
                    {'loss': loss_value.item(), "loss_E": loss_E_value.item()},
                    epoch
                )

            loop.set_description(f"U_p: {field_comp.lmbda}, Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss_value.item(),
                             loss_E=loss_E_value.item() if loss_E_value is not None else None)

            loss_data.append(loss_value.item())
            if intermediateModel_path is not None:
                steps = training_dict["save_model_every_n"]
                if steps > 0 and step_counter >= steps and step_counter % steps == 0:
                    intermModel_path = intermediateModel_path / Path(
                        'intermediate_1NN_' + str(int(field_comp.lmbda * 1000000)) +
                        'by1000000_' + str(step_counter) + '.pt'
                    )
                    torch.save(field_comp.net.state_dict(), intermModel_path)

    return loss_data



def fit_with_early_stopping(field_comp, training_set_collocation, T_conn, area_T, hist_alpha,
                            matprop, pffmodel, weight_decay, num_epochs, optimizer, min_delta,
                            hist_Y_max_over_H=None, hist_alpha_bar=None, intermediateModel_path=None,
                            writer=None, training_dict={}):
    loss_data = list()
    step_counter = 0
    early_stopping = EarlyStopping(tol_steps=10, min_delta=min_delta, device=area_T.device)
    loss_prev = torch.tensor([0.0], device=area_T.device)
    epoch_loss = loss_prev
    
    # Loop over epochs
    for epoch in range(num_epochs):
        loop = tqdm(training_set_collocation, miniters=25)
        # Loop over batches
        for j, (inp_train, outp_train) in enumerate(loop):
            
            optimizer.zero_grad()
            if T_conn is None:
                inp_train.requires_grad = True

            field_outputs = field_comp.fieldCalculation(inp_train)
            u, v, alpha = field_outputs[0], field_outputs[1], field_outputs[2]

            loss_E_el, loss_E_d, loss_hist, _ = compute_energy(
                inp_train, u, v, alpha,
                hist_alpha, matprop, pffmodel,
                area_T, T_conn, hist_Y_max_over_H, hist_alpha_bar
            )

            # Use energy densities so elastic and damage terms are balanced without additional scaling.
            loss_var = loss_E_el + loss_E_d + loss_hist

            # weight regularization
            loss_reg = 0.0
            if weight_decay != 0:
                for name, param in field_comp.net.named_parameters():
                    if 'weight' in name:
                        loss_reg += torch.sum(param**2)

            loss = loss_var + weight_decay * loss_reg

            loss.backward()
            optimizer.step()
            step_counter += 1

            loss_value = loss.detach()
            loss_E_value = loss_var.detach()

            if writer is not None:
                writer.add_scalars(
                    'U_p_' + str(field_comp.lmbda.item()),
                    {'loss': loss_value.item(), "loss_E": loss_E_value.item()},
                    epoch
                )

            loop.set_description(f"U_p: {field_comp.lmbda}, Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss_value.item(), loss_E=loss_E_value.item())

            loss_data.append(loss_value.item())
            if intermediateModel_path is not None:
                steps = training_dict["save_model_every_n"]
                if steps > 0 and step_counter >= steps and step_counter % steps == 0:
                    intermModel_path = intermediateModel_path / Path(
                        'intermediate_1NN_' + str(int(field_comp.lmbda * 1000000)) +
                        'by1000000_' + str(step_counter) + '.pt'
                    )
                    torch.save(field_comp.net.state_dict(), intermModel_path)

            epoch_loss = loss_value

        early_stopping(epoch_loss, loss_prev)
        if early_stopping.early_stop:
            break
        loss_prev = epoch_loss

    return loss_data
