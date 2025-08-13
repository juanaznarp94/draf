from collections import OrderedDict
from typing import Callable
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torchvision.models as models
from torchvision.models import ResNet
from enum import Enum
from torchvision.transforms import RandomAdjustSharpness, GaussianBlur, ColorJitter, ToPILImage, RandomHorizontalFlip
from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter, GaussianBlur, ElasticTransform
import random
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor


class DatasetType(Enum):
    REAL = "real"
    CONCEPT = "concept"
    DATA = "data"


def model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    return model


fds = None


def load_serverData(num_partitions=10):
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(10)

    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    test_loader = DataLoader(partition_train_test["test"], batch_size=64, shuffle=False)
    return train_loader, test_loader


def load_data2(partition_id, num_partitions, strongness) -> tuple[DataLoader, DataLoader]:
    global fds

    if fds is None:
        partitioner = IidPartitioner(num_partitions=10)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    image_manipulation_transforms = []

    if partition_id == 2:
        print(f"Partition {partition_id}: Applying Data Drift.")
        image_manipulation_transforms.extend([
            ElasticTransform(alpha=500.0)
        ])
    else:
        print(f"Partition {partition_id}: No Data Drift applied.")

    final_image_transforms = [
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    image_pipeline = Compose(image_manipulation_transforms + final_image_transforms)

    def apply_image_transforms_to_batch(batch):

        batch["img"] = [image_pipeline(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_image_transforms_to_batch)

    if partition_id == 3:
        print(f"Partition {partition_id}: Applying Concept Drift (label manipulation).")

        def concept_transform_labels(batch):
            num_classes = 10

            batch["label"] = [random.randint(0, num_classes - 1) for _ in range(len(batch["label"]))]
            return batch

        partition_train_test = partition_train_test.with_transform(concept_transform_labels)
    else:
        print(f"Partition {partition_id}: No Concept Drift applied.")

    train_loader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    test_loader = DataLoader(partition_train_test["test"], batch_size=64, shuffle=False)

    return train_loader, test_loader


def load_data(partition_id, num_partitions) -> tuple[DataLoader, DataLoader]:
    global fds

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    image_manipulation_transforms = []

    if partition_id == 2:
        print(f"Partition {partition_id}: Applying Data Drift.")
        image_manipulation_transforms.extend([

            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    else:
        print(f"Partition {partition_id}: No Data Drift applied.")

    final_image_transforms = [
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    image_pipeline = Compose(image_manipulation_transforms + final_image_transforms)

    def apply_image_transforms_to_batch(batch):

        batch["img"] = [image_pipeline(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_image_transforms_to_batch)

    if partition_id == 3:
        print(f"Partition {partition_id}: Applying Concept Drift (label manipulation).")

        def concept_transform_labels(batch):
            num_classes = 10

            batch["label"] = [random.randint(0, num_classes - 1) for _ in range(len(batch["label"]))]
            return batch

        partition_train_test = partition_train_test.with_transform(concept_transform_labels)
    else:
        print(f"Partition {partition_id}: No Concept Drift applied.")

    train_loader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    test_loader = DataLoader(partition_train_test["test"], batch_size=64, shuffle=False)

    return train_loader, test_loader


def train(net, trainloader, epochs, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def validate_model(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    print("in test: " + str(len(testloader)))
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


from torch.utils.data import Subset, DataLoader


def filter_correct_predictions(global_model, train_loader, device):
    global_preds = predict(global_model, train_loader, device)

    true_labels = []
    for batch in train_loader:
        true_labels.extend(batch["label"].tolist())

    correct_indices = [i for i, (p, t) in enumerate(zip(global_preds, true_labels)) if p == t]

    dataset = train_loader.dataset

    subset_dataset = Subset(dataset, correct_indices)

    filtered_loader = DataLoader(subset_dataset, batch_size=train_loader.batch_size, shuffle=False)

    return filtered_loader


def validate_model_for_meta_agg_with_grad_subset(net, testloader, device, max_batches=5):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    correct = 0
    total_samples = 0
    batch_losses = []

    for batch_idx, batch in enumerate(testloader):
        if batch_idx >= max_batches:
            break

        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        outputs = net(images)
        loss_batch = criterion(outputs, labels)

        batch_size = images.size(0)
        batch_losses.append(loss_batch * batch_size)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total_samples += batch_size

    if batch_losses:
        total_loss_val = torch.stack(batch_losses).sum() / total_samples
    else:
        total_loss_val = torch.tensor(0.0, device=device, requires_grad=True)

    accuracy = correct / total_samples if total_samples > 0 else 0.0

    return total_loss_val, accuracy


def predict(net: ResNet, testloader, device):
    net.to(device)
    net.eval()
    predictions = []

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            outputs = net(images)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())

    return predictions


def predict_probs(model, dataloader, device="cpu"):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["img"].to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def count_parameters():
    return sum(p.numel() for p in model().parameters())
