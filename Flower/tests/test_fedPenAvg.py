import numpy as np
import pytest
from unittest.mock import MagicMock
from flwr.common import ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters
from federated_ex.fedPenAvg import FedPenAvg


def create_mock_client(cid):
    client = MagicMock(spec=ClientProxy)
    client.cid = cid
    return client


def create_mock_fitres(shape=(2, 2), num_examples=10):
    weights = [np.ones(shape), np.ones(shape)]
    parameters = ndarrays_to_parameters(weights)
    return FitRes(
        parameters=parameters, num_examples=num_examples, metrics={}, status={}
    )


def test_aggregate_fit_runs_without_error(monkeypatch):
    def dummy_evaluate_fn(server_round, weights, _):
        return 0.5, {}

    strategy = FedPenAvg(device="cpu", evaluate_fn=dummy_evaluate_fn)

    monkeypatch.setattr(
        "federated_ex.fedPenAvg.detect_drift_and_return_penalize",
        lambda *args, **kwargs: (0.1, 0.2),
    )

    client1 = create_mock_client("1")
    client2 = create_mock_client("2")
    results = [
        (client1, create_mock_fitres(num_examples=20)),
        (client2, create_mock_fitres(num_examples=30)),
    ]

    aggregated_params, metrics = strategy.aggregate_fit(
        server_round=3, results=results, failures=[]
    )

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert aggregated_params is not None
