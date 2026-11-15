from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .registry import SPECIES_CLASS_NAMES, package_root


class SpeciesCheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class SpeciesResult:
    class_index: int
    species_id: str
    probabilities: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def create_species_model() -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 1024),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(1024, len(SPECIES_CLASS_NAMES)),
    )
    return model


def validate_species_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    names = tuple(checkpoint.get("class_names", ()))
    if names != SPECIES_CLASS_NAMES:
        raise SpeciesCheckpointError("Species checkpoint class order does not match the expected order")
    if "state_dict" not in checkpoint:
        raise SpeciesCheckpointError("Species checkpoint is missing state_dict")
    return checkpoint


def predict_species(image_path: str | Path, checkpoint_path: str | Path | None = None, device: str = "cpu") -> SpeciesResult:
    checkpoint_path = Path(checkpoint_path or package_root() / "models" / "species" / "best_species.pt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Species checkpoint not found: {checkpoint_path}. Run bacteria-predict train species first.")
    checkpoint = validate_species_checkpoint(checkpoint_path)
    model = create_species_model().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, .456, .406], [.229, .224, .225])])
    with torch.inference_mode():
        probabilities = torch.softmax(model(transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    index = int(np.argmax(probabilities))
    return SpeciesResult(index, SPECIES_CLASS_NAMES[index], tuple(float(item) for item in probabilities))
