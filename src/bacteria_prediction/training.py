from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import SPECIES_CLASS_NAMES
from .species import create_species_model

INITIAL_LR = 0.15
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
MAX_EPOCHS = 300
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8


def _loaders(
    dataset_root: Path,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
):
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    return {
        "train": DataLoader(datasets.ImageFolder(dataset_root / "train", train_transform), batch_size=train_batch_size, shuffle=True, num_workers=0),
        "test": DataLoader(datasets.ImageFolder(dataset_root / "test", eval_transform), batch_size=eval_batch_size, shuffle=False, num_workers=0),
    }


def evaluate_species(dataset_root: str | Path, checkpoint: str | Path, device: str = "cpu") -> dict:
    from .species import validate_species_checkpoint
    checkpoint_data = validate_species_checkpoint(checkpoint)
    model = create_species_model().to(device)
    model.load_state_dict(checkpoint_data["state_dict"])
    model.eval()
    loader = _loaders(Path(dataset_root))["test"]
    predicted, truth = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    return {"accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)), "count": len(truth)}


def train_species(dataset_root: str | Path, output: str | Path, epochs: int = MAX_EPOCHS, batch_size: int = TRAIN_BATCH_SIZE, device: str = "cpu") -> Path:
    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS}")
    loaders = _loaders(Path(dataset_root), train_batch_size=batch_size)
    model = create_species_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_function = torch.nn.CrossEntropyLoss()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(epochs):
        model.train()
        for images, labels in loaders["train"]:
            optimizer.zero_grad()
            loss_function(model(images.to(device)), labels.to(device)).backward()
            optimizer.step()
        scheduler.step()
    torch.save({"state_dict": model.state_dict(), "class_names": list(SPECIES_CLASS_NAMES), "preprocessing": {"size": [224, 224], "mean": [.485, .456, .406], "std": [.229, .224, .225]}}, output)
    return output
