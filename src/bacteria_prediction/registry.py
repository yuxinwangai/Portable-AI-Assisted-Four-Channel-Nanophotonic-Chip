from dataclasses import dataclass
from pathlib import Path

CONCENTRATION_LABELS = tuple(f"10^{index} CFU/mL" for index in range(1, 8))
SPECIES_CLASS_NAMES = ("dachangganjun", "furongxuehujun", "rongzaohujun")


@dataclass(frozen=True)
class SpeciesModel:
    identifier: str
    display_name: str
    config_file: str
    checkpoint_file: str


MODELS = {
    "dachangganjun": SpeciesModel("dachangganjun", "Escherichia coli", "dachangganjun.py", "dachangganjun_epoch_144.pth"),
    "furongxuehujun": SpeciesModel("furongxuehujun", "Vibrio parahaemolyticus", "furongxuehujun.py", "furongxuehujun_epoch_109.pth"),
    "rongzaohujun": SpeciesModel("rongzaohujun", "Vibrio alginolyticus", "rongzaohujun.py", "rongzaohujun_epoch_133.pth"),
}


def get_species(identifier: str) -> SpeciesModel:
    try:
        return MODELS[identifier]
    except KeyError as error:
        choices = ", ".join(MODELS)
        raise ValueError(f"Unknown species {identifier!r}; available values: {choices}") from error


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]
