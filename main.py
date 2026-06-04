import json
import os
import random
import shutil
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets


class Model(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(16 * 8 * 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, height, width = images.shape

        x = images.reshape(batch_size, 1, height, width)

        x = self.conv1(x)  # [B, 8, _, _]
        x = self.dropout(x)
        x = self.conv2(x)  # [B, 16, _, _]
        x = self.dropout(x)
        x = self.activation(x)

        x = self.pool(x)  # [B, 16, 8, 8]
        x = x.reshape(batch_size, 16 * 8 * 8)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)

        return self.linear2(x)  # [B, num_classes]

    def forward_classification(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(images)
        loss = F.cross_entropy(output, targets)
        return loss, output, targets


@dataclass
class ModelConfig:
    num_classes: int
    hidden_size: int
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    model: ModelConfig
    num_epochs: int = 10
    batch_size: int = 128
    num_workers: int = 4
    seed: int = 42
    learning_rate: float = 1.0e-4


@dataclass
class TimeStats:
    min: float
    max: float
    mean: float
    median: float
    samples: int

    @staticmethod
    def stats(samples: list[float]) -> "TimeStats":
        assert len(samples) > 0

        sorted_samples = sorted(samples)

        return TimeStats(
            min=min(samples),
            max=max(samples),
            mean=sum(samples) / len(samples),
            median=sorted_samples[len(samples) // 2],
            samples=len(samples),
        )


def create_artifact_dir(artifact_dir: str) -> None:
    shutil.rmtree(artifact_dir, ignore_errors=True)
    os.makedirs(artifact_dir, exist_ok=True)


def save_config(config: TrainingConfig, artifact_dir: str) -> None:
    config_dict = {
        "model": asdict(config.model),
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
    }
    with open(os.path.join(artifact_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mnist_collate_fn(batch):
    images = []
    targets = []

    for image, label in batch:
        image = torch.as_tensor(np.array(image), dtype=torch.float32)
        image = image.reshape(1, 28, 28)
        image = ((image / 255.0) - 0.1307) / 0.3081
        images.append(image)

        targets.append(torch.tensor(label, dtype=torch.long))

    images = torch.cat(images, dim=0)  # [B, 28, 28]
    targets = torch.stack(targets, dim=0)

    return images, targets


def train(artifact_dir: str, config: TrainingConfig) -> None:
    create_artifact_dir(artifact_dir)
    save_config(config, artifact_dir)

    device = torch.device("cuda")
    seed_everything(config.seed)

    model = Model(
        num_classes=config.model.num_classes,
        hidden_size=config.model.hidden_size,
        dropout=config.model.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    dataset = datasets.MNIST(root="./data", train=True, download=True)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    dataloader_train = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,  # matches `.shuffle(config.seed)`
        num_workers=config.num_workers,
        collate_fn=mnist_collate_fn,
        pin_memory=True,
        generator=generator,
    )

    epoch_samples: list[float] = []
    iter_samples: list[float] = []

    model.train()
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.perf_counter()

        total_loss = 0.0
        iters = 0

        for images, targets in dataloader_train:
            iters += 1
            iter_start = time.perf_counter()

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model(images)
            loss = F.cross_entropy(out, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iter_samples.append(time.perf_counter() - iter_start)

        torch.cuda.synchronize()
        print(f"Epoch: {epoch}, Avg Loss: {total_loss / iters}")
        epoch_samples.append(time.perf_counter() - epoch_start)

    with open(
        os.path.join(artifact_dir, "epoch_timing.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(asdict(TimeStats.stats(epoch_samples)), f, indent=2)

    with open(
        os.path.join(artifact_dir, "iter_timing.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(asdict(TimeStats.stats(iter_samples)), f, indent=2)


if __name__ == "__main__":
    artifact_dir = "./cuda_model"
    config = TrainingConfig(
        model=ModelConfig(num_classes=10, hidden_size=512),
        batch_size=128,
        num_epochs=5,
    )
    train(artifact_dir, config)
