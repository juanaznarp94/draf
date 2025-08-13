import logging
from functools import reduce

import torch
import torch.nn as nn
from torch import optim
from torch.nn import Parameter
from typing import Dict, List, Tuple

from .task import set_weights, get_weights, model, validate_model_for_meta_agg_with_grad_subset
from .setup import loadData
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
import copy
from scipy.optimize import minimize, differential_evolution
import numpy as np


class MetaAggregator(nn.Module):
    def __init__(self, num_clients: int, hidden_dim: int = 256):
        super().__init__()
        self.num_clients = num_clients
        self.coefficients = nn.Parameter(torch.rand((num_clients, 1)))
        self.coefficients_scipy = np.random.uniform(low=0.0, high=1.0, size=(num_clients,))
        self.scipy_bounds = [(0.0, 1.0)] * num_clients

    def forward(self):
        return self.coefficients

    def getScipyParam(self):
        return self.coefficients_scipy


def create_client_models_from_weights(base_model: nn.Module,
                                      client_weights: Dict[str, Parameters],
                                      device: str = 'cpu') -> Dict[str, nn.Module]:
    client_models = {}
    for client_id, weights_params in client_weights.items():
        client_model = copy.deepcopy(base_model).to(device)
        weights_arrays = parameters_to_ndarrays(weights_params)
        set_weights(client_model, weights_arrays)
        client_models[client_id] = client_model
    return client_models


def compute_client_statistics_from_weights(client_models: Dict[str, nn.Module],
                                           global_model: nn.Module,
                                           client_num_examples: Dict[str, int],
                                           client_losses: Dict[str, float],
                                           device: str = 'cpu') -> torch.Tensor:
    client_stats = []

    for client_id, client_model in client_models.items():
        client_model.to(device).eval()
        drift_penalty = compute_reg_term(client_model, global_model)

        stat = torch.tensor(client_losses[client_id], device=device) + drift_penalty
        client_stats.append(stat)

    return torch.stack(client_stats)


def compute_reg_term(client_model, global_model):
    global_params = torch.cat([p.view(-1) for p in global_model.parameters()])
    client_params = torch.cat([p.view(-1) for p in client_model.parameters()])

    l2_drift = torch.sum((client_params - global_params) ** 2)

    return l2_drift / global_params.numel()


def aggregate(results: List[Tuple[NDArrays, int, float]]) -> NDArrays:
    total_weight = sum(num_examples * coeff for (_, num_examples, coeff) in results)

    weighted_weights = [
        [(layer * num_examples * coeff) for layer in weights]
        for weights, num_examples, coeff in results
    ]

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / total_weight
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime


step = 0


def optimize(global_model: nn.Module,
             client_weights_dict: Dict[str, Parameters],
             client_losses_dict: Dict[str, float],
             client_samples_dict: Dict[str, int],
             dataDriftDict: Dict[str, float],
             conceptDriftDict: Dict[str, float],
             server_val_loader,
             device: str = "cpu"):
    client_models = create_client_models_from_weights(model(), client_weights_dict, device=device)

    def objective(coefficients_scipy):
        global step
        results = []
        num_clients = len(coefficients_scipy)

        client_ids = sorted(client_weights_dict.keys())
        for i, (client_id, _) in enumerate(client_weights_dict.items()):
            params = parameters_to_ndarrays(client_weights_dict[client_id])

            num_samples = client_samples_dict[client_id]
            coeff = coefficients_scipy[i]
            results.append((params, num_samples, coeff))

        aggregate_param = aggregate(results)
        set_weights(global_model, aggregate_param)

        global_model.eval()

        val_loss, acc = validate_model_for_meta_agg_with_grad_subset(global_model, server_val_loader, device,
                                                                     max_batches=32)

        total_reg_term = 0.0
        for i, (client_id, _) in enumerate(client_weights_dict.items()):
            client_loss = client_losses_dict[client_id]
            dataDrift = dataDriftDict[client_id]
            conceptDrift = conceptDriftDict[client_id]

            lambda_scale = min(1.0, dataDrift + conceptDrift)
            regterm = compute_reg_term(global_model=global_model, client_model=client_models[client_id]).item()

            total_reg_term += client_loss + (lambda_scale * regterm)
        logging.error(
            f"[Optimize][Step][{step + 1}] Coeff:{coefficients_scipy} ValLoss: {val_loss}, RegTerm: {total_reg_term}")
        step += 1
        return (1.0 - acc) + (total_reg_term / num_clients)

    return objective


def server_aggregate_round(meta_aggregator: MetaAggregator,
                           global_model: nn.Module,
                           client_weights_dict: Dict[str, Parameters],
                           client_losses_dict: Dict[str, float],
                           client_samples_dict: Dict[str, int],
                           dataDriftDict: Dict[str, float],
                           conceptDriftDict: Dict[str, float],
                           lr: float = 0.001,
                           steps: int = 10) -> Parameters:
    server_val_loader, _, _ = loadData(0, 1, 32)

    objective_func = optimize(global_model,
                              client_weights_dict,
                              client_losses_dict,
                              client_samples_dict, dataDriftDict, conceptDriftDict,
                              server_val_loader,
                              device="cpu")
    result = differential_evolution(
        objective_func,
        bounds=meta_aggregator.scipy_bounds,
        maxiter=steps,
        strategy="best1bin",
        polish=True, popsize=5,
        x0=meta_aggregator.coefficients_scipy
    )

    meta_aggregator.coefficients_scipy = result.x

    return ndarrays_to_parameters(get_weights(global_model))
