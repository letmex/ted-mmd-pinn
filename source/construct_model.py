import torch
from pff_model import PFFModel
from material_properties import MaterialProperties
from network import NeuralNet, init_xavier

def construct_model(PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, device):
    # Phase field model
    pffmodel = PFFModel(PFF_model = PFF_model_dict["PFF_model"], 
                        se_split = PFF_model_dict["se_split"],
                        tol_ir = torch.tensor(PFF_model_dict["tol_ir"], device=device))

    # Material model
    def _get_tensor(key):
        value = mat_prop_dict.get(key, None)
        if value is None:
            return None
        return torch.tensor(value, device=device)

    matprop = MaterialProperties(mat_E = torch.tensor(mat_prop_dict["mat_E"], device=device), 
                                mat_nu = torch.tensor(mat_prop_dict["mat_nu"], device=device), 
                                w1 = torch.tensor(mat_prop_dict["w1"], device=device), 
                                l0 = torch.tensor(mat_prop_dict["l0"], device=device),
                                ft = _get_tensor("ft"),
                                Gf0 = _get_tensor("Gf0"),
                                b0 = _get_tensor("b0"),
                                xi = _get_tensor("xi"),
                                c0 = _get_tensor("c0"),
                                p = _get_tensor("p"),
                                E_bar = _get_tensor("E_bar"),
                                lch_bar = _get_tensor("lch_bar"))

    # Neural network
    torch.manual_seed(network_dict["seed"])

    output_dimension = network_dict.get("output_dimension", domain_extrema.shape[0] + 1)
    network = NeuralNet(input_dimension=domain_extrema.shape[0], 
                        output_dimension=output_dimension,
                        n_hidden_layers=network_dict["hidden_layers"],
                        neurons=network_dict["neurons"],
                        activation=network_dict["activation"],
                        init_coeff=network_dict["init_coeff"])
    init_xavier(network)

    return pffmodel, matprop, network
