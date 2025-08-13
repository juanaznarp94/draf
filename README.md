# DRAF: A Drift-Resilient Averaging in Federated Learning

## Setup

# federated-ex: A Flower / PyTorch app

## Execution on AWS via terraform

1. Create a .env file
2. Copy from the content from the .env.example file into your .env file
3. Fill out the file
4. Choose strategy: Either ``FedAvg`` or ``custom`` (our FedPenAvg strategy)
5. Choose the level of data and client drift: ``NO``, ``WEAK``, ``MIDDLE`` or ``STRONG``
6. 
Run Docker Compose from the root of the project:

```bash
docker-compose up --build

```
7. Open a terminal inside the docker container and run 

```bash
#Inside Docker container
sh ./deploy.sh
```
to deploy the setup on AWS

8. To tear everything down:

```bash
#Inside Docker container /workspace
sh ./destroy.sh
```

After that you can inspect the training reslults in AWS CloudWatch.
Optional
---
### SSH into Device EC2s

For each client and the server, a corresponding SSH script is generated to connect with the EC2  in:

```bash
#Inside Docker container
/workspace/ssh-scripts/
```

To connect to the server instance, inside the container:

```bash
#Inside Docker container
cd ssh-scripts
bash ./serverId_0.sh 
```

To connect to a client instance, inside the container:

```bash
#Inside Docker container
cd ssh-scripts
bash ./clientId_1.sh 
```
Within the server and client EC2's you can inspect the logs of the scripts by:
1. Coping the **containerID** which yu can get with
```bash
docker ps -a 
```
2. After that you can inspect the logs by 
```bash
docker logs -f <containerID>
```
**NOTE!!**
Some logs are declared as ERROR but these are nor Errors.
We used logging.error because we had a bug with Flowers. When we used logging.info it didn't
appear. 

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
