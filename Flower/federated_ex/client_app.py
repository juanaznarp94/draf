import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federated_ex.task import (
    get_weights,
    set_weights,
    validate_model,
    train,
    DatasetType,
    model,
)
from federated_ex.setup import loadData


class FlowerClient(NumPyClient):
    def __init__(self, net, partition_id, run_id, local_epochs):
        self.net = net
        self.partition_id = partition_id
        self.run_id = run_id
        self.local_epochs = local_epochs
        self.device = torch.device("cpu")
        self.net.to(self.device)
        self.client_id = client_id
        self.run_id = run_id
        self.isManipulated = isManipulated
        self.manipulateType = manipulateType

        self.client_logger = Logger(run_id=self.run_id, client_id=str(self.client_id))

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        server_round = config["server_round"]

        trainloader, _, metadata = loadData(
            client=self.partition_id,
            round=server_round
        )
        print("In Training before Train")
        train_loss = train(
            self.net,
            trainloader,
            self.local_epochs,
            self.device,
        )
        print("After training")
        return (
            get_weights(self.net),
            len(trainloader.dataset),
            {
                "train_loss": train_loss,
                "partition_id": self.partition_id,
                "drift": metadata["drift"],
                "strongness": metadata["strongness"]
            },
        )

    def evaluate(self, parameters, config):
        server_round = config.get("server_round")

        set_weights(self.net, parameters)

        server_round = config["server_round"]
        _, valloader, metadata = loadData(
            client=self.partition_id,
            round=server_round
        )

        loss, accuracy = validate_model(self.net, valloader, self.device)
        return loss, len(valloader.dataset), {"accuracy": accuracy, "partition_id": self.partition_id,
                                              "drift": metadata["drift"], "strongness": metadata["strongness"]}

        client_metric_entry = ClientMetrics()
        client_metric_entry.round = server_round
        client_metric_entry.client_id = int(self.client_id)
        client_metric_entry.isManipulated = self.isManipulated
        client_metric_entry.maniPulateType = self.manipulateType
        client_metric_entry.client_loss = float(loss)
        client_metric_entry.client_accuracy = float(accuracy)
        client_metric_entry.num_examples = len(self.valloader.dataset)
        self.client_logger.addClientMetric(client_metric_entry)

        return loss, len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    net = model()
    partition_id = context.node_config["partition-id"]
    local_epochs = context.run_config["local-epochs"]
    run_id = context.run_id
    return FlowerClient(
        net=net,
        partition_id=partition_id,
        run_id=run_id,
        local_epochs=local_epochs
    ).to_client()

    client_id = os.environ.get("CLIENT", f"client-{partition_id}")
    run_id = context.run_id if hasattr(context, 'run_id') else os.environ.get("FL_RUN_ID",
                                                                              datetime.now().strftime("%Y%m%d%H%M%S"))


app = ClientApp(
    client_fn,
)
