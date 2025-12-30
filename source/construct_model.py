import torch
from pff_model import PFFModel
from material_properties import MaterialProperties
from network import NeuralNet, init_xavier

def construct_model(PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, device):
    # Phase field model
    pffmodel = PFFModel(
        PFF_model=PFF_model_dict.get("PFF_model", "AT1"),
        se_split=PFF_model_dict.get("se_split", "volumetric"),
        tol_ir=torch.tensor(PFF_model_dict.get("tol_ir", 5e-3), device=device),
        c0=mat_prop_dict.get("c0", 1.0),
        l0=torch.tensor(mat_prop_dict.get("l0", 0.01), device=device),
        xi=mat_prop_dict.get("xi", 1.0),
        eta_S=mat_prop_dict.get("eta_S", 1.0),
        alpha_T=mat_prop_dict.get("alpha_T", 1.0),
        p_fatigue=mat_prop_dict.get("p_fatigue", 1.0),
        G_f0=mat_prop_dict.get("Gf0", 1.0),
        k_T=PFF_model_dict.get("k_T", 0.0),
        theta_0=PFF_model_dict.get("theta_0", 0.0),
    )

    # Material model
    matprop = MaterialProperties(
        mat_E=torch.tensor(mat_prop_dict["mat_E"], device=device),
        mat_nu=torch.tensor(mat_prop_dict["mat_nu"], device=device),
        w1=torch.tensor(mat_prop_dict["w1"], device=device),
        l0=torch.tensor(mat_prop_dict["l0"], device=device),
        ft=mat_prop_dict.get("ft"),
        Gf0=mat_prop_dict.get("Gf0"),
        b0=mat_prop_dict.get("b0"),
        xi=mat_prop_dict.get("xi"),
        c0=mat_prop_dict.get("c0"),
        p=mat_prop_dict.get("p"),
        E_bar=mat_prop_dict.get("E_bar"),
        lch_bar=mat_prop_dict.get("lch_bar"),
        thermal_props=mat_prop_dict.get("thermal_props"),
    )

    # Neural network
    torch.manual_seed(network_dict["seed"])
    output_dimension = network_dict.get("output_dimension", domain_extrema.shape[0] + 1)
    network = NeuralNet(
        input_dimension=domain_extrema.shape[0],
        output_dimension=output_dimension,
        n_hidden_layers=network_dict["hidden_layers"],
        neurons=network_dict["neurons"],
        activation=network_dict["activation"],
        init_coeff=network_dict["init_coeff"],
    )
    init_xavier(network)

    return pffmodel, matprop, network
