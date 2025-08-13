from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_ex.task import get_weights, set_weights, validate_model, model, count_parameters
from flwr.common import Context
from torch.utils.data import DataLoader
import torch
from federated_ex.logger import Logger
from federated_ex.fedPenAvg import FedPenAvg
import os
import datetime
from dotenv import load_dotenv
from federated_ex.setup import loadData, main
from federated_ex.metaAggr import MetaAggregator

import os, datetime
import torch
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from federated_ex.task import get_weights, set_weights, test, load_data, model

load_dotenv()


def get_fit_config_fn(strategy_type: str):
    def fit_config(server_round: int) -> dict:
        return {
            "server_round": server_round,
            "strategy_type": strategy_type,
        }

    return fit_config


def gen_evaluate_fn(
        testloader: DataLoader,
        device: torch.device,
):
    def evaluate(server_round, parameters_ndarrays, config):
        net = model()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = validate_model(net, testloader, device=device)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_config_fn(strategy_type: str):
    def evaluate_config(server_round: int) -> dict:
        return {
            "server_round": server_round,
            "strategy_type": strategy_type,
        }

    return evaluate_config


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    strategy_type = context.run_config["strategy_type"]
    min_available_clients = context.run_config["min_available_clients"]
    ndarrays = get_weights(model())
    parameters = ndarrays_to_parameters(ndarrays)
    aws_secret_key = context.run_config["aws_secret_key"]
    aws_token = context.run_config["aws_token"]
    aws_access_key = context.run_config["aws_access_key"]
    aws_region = context.run_config["aws_region"]
    run_name = context.run_config["run_name"]
    trainloader, testloader, _ = loadData(0, 1, 32)
    metaAggr = MetaAggregator(num_clients=min_available_clients)
    SERVER_RUN_ID = run_name
    logger = Logger(run_id=SERVER_RUN_ID,
                    region_name=aws_region,
                    aws_session_token=aws_token,
                    aws_secret_key=aws_secret_key,
                    aws_access_key=aws_access_key
                    )

    strategy = FedPenAvg(
        metagg=metaAggr,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        min_fit_clients=min_available_clients,
        initial_parameters=parameters,
        evaluate_fn=gen_evaluate_fn(testloader, device="cpu"),
        evaluate_metrics_aggregation_fn=weighted_average,
        device="cpu",
        on_evaluate_config_fn=get_evaluate_config_fn(strategy_type=strategy_type),
        logger_instance=logger,
        strategy_type=strategy_type,
        on_fit_config_fn=get_fit_config_fn(strategy_type=strategy_type)

    )

    SERVER_RUN_ID = context.run_id if hasattr(context, 'run_id') else os.environ.get("FL_RUN_ID",
                                                                                     datetime.now().strftime(
                                                                                         "%Y%m%d%H%M%S"))
    my_server_logger = Logger(run_id=SERVER_RUN_ID)

    print(f"Using strategy: {strategy_type}")
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
