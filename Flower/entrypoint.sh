#!/bin/sh
set -e
set -x
python initToml.py
. ./.env
if [ "${TYPE}" = "server" ]; then
    echo "Starting server..."
    python federated_ex/setup.py
    flower-superlink --insecure &
    exec flwr run . local-deployment --stream
fi

if [ "${TYPE}" = "client" ]; then
    echo "Starting client with CLIENT=$CLIENT"
    exec flower-supernode --insecure \
        --superlink "${SERVER_IP}:9092" \
        --clientappio-api-address 127.0.0.1:9097 \
        --node-config "partition-id=$CLIENT"
fi

