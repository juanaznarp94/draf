import os
import sys
import logging
import os
import json
from dotenv import load_dotenv
import uuid

load_dotenv()
def main():
    num_server_rounds = int(os.getenv("NUM_SERVER_ROUNDS"))
    fraction_fit = int(os.getenv("FRACTION_FIT"))
    local_epochs = int(os.getenv("LOCAL_EPOCHS"))
    strategy = os.getenv("STRATEGY")
    min_clients = int(os.getenv("MIN_CLIENTS"))

    ipAddr = os.getenv("SERVER_IP")

    with open("pyproject.toml", "w+") as file:
        file.write(
            f"""
    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    [project]
    name = "federated_ex"
    version = "1.0.0"
    description = ""
    license = "Apache-2.0"
    dependencies = [
    "flwr[simulation]>=1.18.0",
    "flwr-datasets[vision]>=0.5.0",
    "torch==2.5.1+cpu",
    "torchvision==0.20.1+cpu",
    "scipy",
    "boto3",
    "watchtower"
]


    [tool.hatch.build.targets.wheel]
    packages = ["."]

    [tool.flwr.app]
    publisher = "JonaMarco"

    [tool.flwr.app.components]
    serverapp = "federated_ex.server_app:app"
    clientapp = "federated_ex.client_app:app"

    [tool.flwr.app.config]
    num-server-rounds = {num_server_rounds}
    fraction-fit = {fraction_fit}
    local-epochs = {local_epochs}
    strategy_type = "{strategy}"
    min_available_clients = {min_clients}
    aws_secret_key = "{os.getenv("AWS_SECRET_ACCESS_KEY")}"
    aws_token = "{os.getenv("AWS_SESSION_TOKEN")}"
    aws_access_key = "{os.getenv("AWS_ACCESS_KEY_ID")}"
    aws_region = "{os.getenv("AWS_REGION")}"
    run_name = "{os.getenv("RUN_NAME")}"
    
    [tool.flwr.federations.local-deployment]
    address = "{ipAddr}:9093"
    insecure = true
    """
        )


if __name__ == "__main__":
    main()
