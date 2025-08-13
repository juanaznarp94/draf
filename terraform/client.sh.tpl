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



cd /home/ec2-user/Flower/
python initToml.py ${num_partitions} ${num_server_rounds} ${fraction_fit} ${local_epochs} ${strategy} ${strategy} ${min_clients} "client" ${serverIp}
pip install -e .
cd /home/ec2-user/


cd /home/ec2-user/Flower/
flower-supernode \
--insecure \
--superlink ${serverIp}:9092 \
--clientappio-api-address 127.0.0.1:9094 \
--node-config "partition-id="${client} "num-partitions=${num_partitions}"
