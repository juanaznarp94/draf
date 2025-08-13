from unittest.mock import MagicMock
from federated_ex.client_app import client_fn, FlowerClient
from federated_ex.task import model, get_weights, load_data


def create_mock_context():
    context = MagicMock()
    context.node_config = {
        "partition-id": 0,
        "num-partitions": 3,
        "manipulateType": "real",
    }
    context.run_config = {
        "local-epochs": 1,
    }
    return context


def test_flower_client_fit_and_evaluate():
    net = model()
    trainloader, testloader = load_data(1, 100)
    client = FlowerClient(net, trainloader, testloader, local_epochs=1)

    parameters = get_weights(net)
    new_params, num_examples, metrics = client.fit(parameters, config={})

    assert isinstance(new_params, list)
    assert isinstance(num_examples, int)
    assert "train_loss" in metrics

    loss, num_eval_examples, eval_metrics = client.evaluate(parameters, config={})
    assert isinstance(loss, float)
    assert "accuracy" in eval_metrics


def test_client_fn_returns_flower_client():
    context = create_mock_context()
    flwr_client = client_fn(context)

    assert hasattr(flwr_client, "fit")
    assert hasattr(flwr_client, "evaluate")
