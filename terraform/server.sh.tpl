#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1


sudo amazon-linux-extras enable python3.8
sudo yum install -y python3.8
sudo alternatives --install /usr/bin/python python /usr/bin/python3.8 1
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
mkdir -p /home/ec2-user

mkdir -p /home/ec2-user/Flower
mkdir -p /home/ec2-user/Flower/federated_ex
cp "${flowerDir}"/federated_ex/__init__.py /home/ec2-user/Flower/federated_ex/__init__.py
echo "${flowerDir}"/federated_ex/__init__.py | base64 -d > /home/ec2-user/Flower/federated_ex/__init__.py
echo "${flowerDir}"/federated_ex/client_app.py | base64 -d > /home/ec2-user/Flower/federated_ex/client_app.py
echo "${flowerDir}"/federated_ex/driftDetection.py | base64 -d > /home/ec2-user/Flower/federated_ex/driftDetection.py
echo "${flowerDir}"/federated_ex/fedPenAvg.py | base64 -d > /home/ec2-user/Flower/federated_ex/fedPenAvg.py
echo "${flowerDir}"/federated_ex/server_app.py | base64 -d > /home/ec2-user/Flower/federated_ex/server_app.py
echo "${flowerDir}"/federated_ex/task.py | base64 -d > /home/ec2-user/Flower/federated_ex/task.py

echo "${flowerDir}"/pyproject.toml | base64 -d > /home/ec2-user/Flower/pyproject.toml
echo "${flowerDir}"/initToml.py | base64 -d > /home/ec2-user/Flower/initToml.py
echo "${flowerDir}"/__init__.py | base64 -d > /home/ec2-user/Flower/__init__.py


cd /home/ec2-user/Flower/
python initToml.py ${num_partitions} ${num_server_rounds} ${fraction_fit} ${local_epochs} ${strategy} ${strategy} ${min_clients} "server"
pip install -e .
cd /home/ec2-user/


cd /home/ec2-user/Flower/
flower-superlink --insecure
flwr run . local-deployment --stream
