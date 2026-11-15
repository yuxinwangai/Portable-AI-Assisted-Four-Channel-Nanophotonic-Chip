from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .registry import CONCENTRATION_LABELS, get_species, package_root


@dataclass(frozen=True)
class ConcentrationResult:
    class_index: int
    label: str
    probabilities: tuple[float, ...]
    log10_cfu_ml: float
    cfu_ml: float

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_probabilities(probability_rows: np.ndarray) -> ConcentrationResult:
    rows = np.asarray(probability_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError("Probability vectors must be a two-dimensional N by seven (7) array")
    if not np.all(np.isfinite(rows)) or np.any(rows < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    sums = rows.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise ValueError("Each probability vector must sum to 1")
    probabilities = rows.mean(axis=0)
    probabilities /= probabilities.sum()
    class_index = int(np.argmax(probabilities))
    log10_value = float(np.dot(probabilities, np.arange(1, 8)))
    return ConcentrationResult(class_index, CONCENTRATION_LABELS[class_index], tuple(float(x) for x in probabilities), log10_value, float(10 ** log10_value))


def _largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise ValueError("No valid foreground region detected")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8) * 255


def segment_image(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, coarse = cv2.threshold(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(cv2.bitwise_not(coarse), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Segmentation failed: no contours found")
    mask = np.zeros_like(coarse)
    cv2.fillPoly(mask, contours, 255)
    mask = _largest_component(mask)
    ys, xs = np.where(mask == 255)
    if not len(xs):
        raise ValueError("Segmentation failed: empty foreground")
    crop = (bgr * (mask[..., None] == 255))[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, fine = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fine = cv2.medianBlur(fine, 31)
    ys, xs = np.where(fine == 255)
    if not len(xs):
        raise ValueError("Segmentation failed: empty refined foreground")
    return crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def extract_valid_patches(bgr: np.ndarray, patch_size: int = 512, stride: int = 128) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    height, width = bgr.shape[:2]
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            patch = bgr[y:min(y + patch_size, height), x:min(x + patch_size, width)]
            if not patch.size or (patch_size * patch_size - patch.shape[0] * patch.shape[1]) / patch_size ** 2 > .5:
                continue
            patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)), constant_values=0)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if np.mean(gray == 0) > .8:
                continue
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = cv2.medianBlur(binary, 5)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if count <= 1:
                continue
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[label, cv2.CC_STAT_AREA])
            contours, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(contours[0])
            short = min(rw, rh)
            if short and max(rw, rh) / short > 5 and 500 < area < 10000:
                patches.append(patch)
    return patches


def predict_concentration(image_path: str | Path, species: str, input_mode: str = "raw", device: str = "cpu", batch_size: int = 16) -> ConcentrationResult:

    if input_mode not in {"raw", "segmented", "patch"}:
        raise ValueError("input_mode must be raw, segmented, or patch")
    model = get_species(species)
    root = package_root()
    config = root / "configs" / "concentration" / model.config_file
    checkpoint = root / "models" / "concentration" / model.checkpoint_file
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing concentration model weights: {checkpoint}")
    image = np.asarray(Image.open(image_path).convert("RGB"))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if input_mode == "patch" else (segment_image(image) if input_mode == "raw" else cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    patches = [bgr] if input_mode == "patch" else extract_valid_patches(bgr)
    if not patches:
        raise ValueError("No patches passed quality filtering; concentration prediction is unavailable")
    from mmpretrain import ImageClassificationInferencer
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index, patch in enumerate(patches):
            path = Path(directory) / f"patch_{index}.png"
            cv2.imencode(".png", patch)[1].tofile(str(path))
            paths.append(str(path))
        inferencer = ImageClassificationInferencer(str(config), pretrained=str(checkpoint), device=device)
        scores = [item["pred_scores"] for item in inferencer(paths, batch_size=batch_size)]
    return aggregate_probabilities(np.asarray(scores))
