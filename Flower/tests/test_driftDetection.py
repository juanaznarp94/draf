import pytest
from federated_ex.driftDetection import (
    detect_drift_and_return_penalize,
    chi_square_test,
    detect_data_drift,
    detect_concept_drift,
)
from federated_ex.task import model, get_weights, load_serverData
from flwr.common import ndarrays_to_parameters
import torch


def test_detect_drift_returns_two_floats():
    client_model = model()
    global_model = model()
    client_params = ndarrays_to_parameters(get_weights(client_model))
    global_params = ndarrays_to_parameters(get_weights(global_model))

    drift = detect_drift_and_return_penalize(
        cid="test",
        client_parameters=client_params,
        global_params=global_params,
        device="cpu",
        server_round=3,
    )

    assert isinstance(drift, tuple)
    assert len(drift) == 2
    assert all(isinstance(x, float) for x in drift)


def test_chi_square_test_basic():
    pred1 = [0, 1, 2, 0, 1, 2]
    pred2 = [0, 1, 2, 0, 1, 2]
    result = chi_square_test(pred1, pred2)

    assert isinstance(result, float)
    assert result >= 0


def test_chi_square_test_with_drift():
    pred1 = [0, 0, 0, 0, 0, 0]
    pred2 = [1, 1, 1, 1, 1, 1]
    result = chi_square_test(pred1, pred2)

    assert isinstance(result, float)
    assert result > 0


def test_chi_square_test_edge_case():
    pred1 = [1, 1, 1]
    pred2 = [1, 1, 1]
    result = chi_square_test(pred1, pred2)

    assert result == 1


def test_detect_concept_drift_returns_float():
    device = "cpu"
    model1 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    model2 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))

    test_loader = load_serverData()

    result = detect_concept_drift(model1, model2, test_loader, device)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_detect_data_drift_returns_float():
    device = "cpu"
    model1 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    model2 = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    test_loader = load_serverData()
    result = detect_concept_drift(model1, model2, test_loader, device)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
