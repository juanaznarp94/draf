import os
import random
import json
import tempfile
import zipfile
import boto3
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter, GaussianBlur, ElasticTransform
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from PIL import Image
import numpy as np
import torch

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
region_name = os.getenv("AWS_REGION")

concept = {"WEAK": 0.1, "MIDDLE": 0.5, "STRONG": 1.0}
data_drift = {"WEAK": 200.0, "MIDDLE": 350.0, "STRONG": 500.0}


def getBucketPrefix(runId, client, round):
    return f"run_{runId}/round_{round}/client_{client}"


def conceptDriftTransform(strongness: str, dataset):
    severity = concept[strongness]

    def flip_label(batch):
        num_to_flip = int(len(batch["label"]) * severity)
        new_labels = batch["label"].copy()

        indices_to_flip = random.sample(range(len(batch["label"])), num_to_flip)

        for idx in indices_to_flip:
            original_label = new_labels[idx]
            new_label = (original_label + 1) % 10
            new_labels[idx] = new_label

        batch["label"] = new_labels
        return batch

    return dataset.map(flip_label, batched=True)


def dataDriftTransform(strongness: str):
    severity = data_drift[strongness]
    elastic = ElasticTransform(alpha=severity)
    pytorch_transforms = Compose([elastic, ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    return apply_transforms


def defaultTransform():
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    return apply_transforms


def uploadS3(path_prefix: str, metadata: dict, dataset=None):
    s3 = boto3.client("s3",
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      aws_session_token=aws_session_token,
                      region_name=region_name)
    bucket = os.getenv("BUCKET")

    metadata_key = f"{path_prefix}/metadata.json"
    dataset_key = f"{path_prefix}/dataset.zip"

    try:
        s3.head_object(Bucket=bucket, Key=metadata_key)
        s3.head_object(Bucket=bucket, Key=dataset_key)
        print(f"Skipped upload for {path_prefix} (already exists)")
        return
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise

    s3.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata).encode("utf-8")
    )
    print(f"Uploaded metadata to s3://{bucket}/{metadata_key}")

    if dataset is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = os.path.join(tmp_dir, "dataset")
            dataset.save_to_disk(dataset_path)

            zip_path = os.path.join(tmp_dir, "dataset.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(dataset_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, dataset_path)
                        zipf.write(file_path, arcname)

            with open(zip_path, "rb") as f:
                s3.upload_fileobj(f, bucket, dataset_key)
            print(f"Uploaded dataset to s3://{bucket}/{dataset_key}")


def loadData(client, round, batch_size=32):
    runId = os.getenv("RUN_NAME")
    s3 = boto3.client("s3",
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      aws_session_token=aws_session_token,
                      region_name=region_name)
    bucket = os.getenv("BUCKET")

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = f"run_{runId}/round_{round}/client_{client}"
        metadata_key = f"{prefix}/metadata.json"
        zip_key = f"{prefix}/dataset.zip"

        metadata_path = os.path.join(tmpdir, "metadata.json")
        zip_path = os.path.join(tmpdir, "dataset.zip")
        dataset_dir = os.path.join(tmpdir, "dataset")

        s3.download_file(bucket, metadata_key, metadata_path)
        s3.download_file(bucket, zip_key, zip_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        dataset = load_from_disk(dataset_dir)

        drift_type = metadata.get("drift", "NO")
        strongness = metadata.get("strongness", "WEAK")

        if drift_type == "DataDrift":
            print(f"Applying DataDrift for client {client}, round {round} with strongness {strongness}")
            dataset = dataset.with_transform(dataDriftTransform(strongness))
        if drift_type == "ConceptDrift":
            print(f"Applying ConceptDrift for client {client}, round {round} with strongness {strongness}")

            dataset = conceptDriftTransform(strongness, dataset)
            dataset = dataset.with_transform(defaultTransform())

        if drift_type not in ["DataDrift", "DataDriftAndConceptDrift", "ConceptDrift"]:
            dataset = dataset.with_transform(defaultTransform())

        print("Transformed")
        train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True if client != 0 else False)
        test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)

        print("Splitted")
        return train_loader, test_loader, metadata


def main():
    runId = os.getenv("RUN_NAME")
    clients = int(os.getenv("MIN_CLIENTS")) + 1
    rounds = int(os.getenv("NUM_SERVER_ROUNDS"))
    strongness = os.getenv("STRONGNESS")

    partitioner = IidPartitioner(num_partitions=clients)
    fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})

    data_drifted_clients = []
    concept_drift_client = []

    if strongness != "NO":
        i = 0
        while i < 3:
            dataDriftClient = random.randrange(1, clients)
            if dataDriftClient not in data_drifted_clients:
                data_drifted_clients.append(dataDriftClient)
                i += 1
        i = 0
        while i < 2:
            conceptDriftClient = random.randrange(1, clients)
            if (conceptDriftClient in data_drifted_clients) or (conceptDriftClient in concept_drift_client):
                continue
            concept_drift_client.append(conceptDriftClient)
            i += 1

    for round in range(1, rounds + 1):

        for client in range(clients):
            partition = fds.load_partition(client)

            dataset = partition.train_test_split(test_size=0.2, seed=42)

            drift_type_for_metadata = "NO"

            if client in data_drifted_clients:
                drift_type_for_metadata = "DataDrift"
            if client in concept_drift_client:
                drift_type_for_metadata = "ConceptDrift"

            print(
                f"Uploading plain dataset for client {client}, round {round}. Intended drift: {drift_type_for_metadata}")
            uploadS3(
                getBucketPrefix(runId, client, round),
                metadata={"drift": drift_type_for_metadata, "strongness": strongness},
                dataset=dataset
            )


if __name__ == '__main__':
    main()
