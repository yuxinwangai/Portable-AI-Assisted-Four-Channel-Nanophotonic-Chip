import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from .registry import SPECIES_CLASS_NAMES


def original_image_key(filename: str) -> str:

    return re.sub(r"_\d+_\d+(?:_\d+)?(?=\.[^.]+$)", "", filename)


def make_stratified_split(items: list[tuple[str, str]], seed: int = 42) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for species, filename in items:
        if species not in SPECIES_CLASS_NAMES:
            raise ValueError(f"Invalid species: {species}")
        grouped[species][original_image_key(filename)].append(filename)
    output = {"train": [], "test": []}
    randomizer = random.Random(seed)
    for species, originals in grouped.items():
        groups = list(originals.values())
        randomizer.shuffle(groups)
        count = len(groups)
        if count < 2:
            raise ValueError(f"{species} requires at least two independent source images for an 80:20 split")
        train_count = min(count - 1, max(1, round(count * .80)))
        for index, group in enumerate(groups):
            partition = "train" if index < train_count else "test"
            output[partition].extend((species, name) for name in group)
    return output


def prepare_species_dataset(source_root: str | Path, destination_root: str | Path, seed: int = 42) -> dict[str, int]:
    source_root, destination_root = Path(source_root), Path(destination_root)
    items: list[tuple[str, str]] = []
    for species in SPECIES_CLASS_NAMES:
        directory = source_root / species
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {directory}")
        items.extend((species, path.name) for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif"})
    split = make_stratified_split(items, seed)
    if not split["test"]:
        raise ValueError("Each class requires enough independent source images to create a test set")
    for partition, entries in split.items():
        for species, name in entries:
            target = destination_root / partition / species / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / species / name, target)
    return {partition: len(entries) for partition, entries in split.items()}
