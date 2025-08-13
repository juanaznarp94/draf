from federated_ex.server_app import server_fn
from flwr.common import Context
from flwr.server.strategy import FedAvg
from federated_ex.fedPenAvg import FedPenAvg
from flwr.server import ServerAppComponents


def test_server_fn_returns_fedavg():
    context = Context(
        run_id="test_run",
        node_id="test_node",
        state="INIT",
        run_config={
            "num-server-rounds": 3,
            "fraction-fit": 0.5,
            "strategy_type": "FedAvg",
            "min_available_clients": 5
        },
        node_config={},
    )

    components = server_fn(context)
    assert isinstance(components, ServerAppComponents)
    assert isinstance(components.strategy, FedAvg)
    assert components.config.num_rounds == 3


def test_server_fn_returns_fedpenavg():
    context = Context(
        run_id="test_run",
        node_id="test_node",
        state="INIT",
        run_config={
            "num-server-rounds": 5,
            "fraction-fit": 0.4,
            "strategy_type": "custom",
            "min_available_clients": 3,
        },
        node_config={},
    )

    components = server_fn(context)
    assert isinstance(components, ServerAppComponents)
    assert isinstance(components.strategy, FedPenAvg)
    assert components.config.num_rounds == 5


def test_server_fn_invalid_strategy_raises():
    context = Context(
        run_id="test_run",
        node_id="test_node",
        state="INIT",
        run_config={
            "num-server-rounds": 2,
            "fraction-fit": 0.2,
            "strategy_type": "invalid",
            "min_available_clients": 2,
        },
        node_config={},
    )

    try:
        server_fn(context)
    except ValueError as e:
        assert "Unknown strategy_type" in str(e)
    else:
        assert False, "Expected ValueError was not raised"
