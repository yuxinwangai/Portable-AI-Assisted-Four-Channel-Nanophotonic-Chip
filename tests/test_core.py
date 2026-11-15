import math

import numpy as np

from bacteria_prediction.concentration import aggregate_probabilities
from bacteria_prediction.registry import CONCENTRATION_LABELS, get_species
from bacteria_prediction.species import SpeciesCheckpointError, create_species_model, validate_species_checkpoint
from bacteria_prediction.data import make_stratified_split
from bacteria_prediction.cli import write_results
from bacteria_prediction.training import EVAL_BATCH_SIZE, INITIAL_LR, MAX_EPOCHS, MOMENTUM, TRAIN_BATCH_SIZE, WEIGHT_DECAY


def test_registry_exposes_all_three_species_and_seven_concentration_labels():
    assert get_species("dachangganjun").display_name == "Escherichia coli"
    assert get_species("furongxuehujun").display_name == "Vibrio parahaemolyticus"
    assert get_species("rongzaohujun").display_name == "Vibrio alginolyticus"
    assert CONCENTRATION_LABELS == tuple(f"10^{index} CFU/mL" for index in range(1, 8))


def test_species_model_and_training_defaults_match_protocol():
    model = create_species_model()

    assert model.fc[0].in_features == 2048
    assert model.fc[0].out_features == 1024
    assert model.fc[2].p == 0.5
    assert model.fc[3].out_features == 3
    assert INITIAL_LR == 0.15
    assert MOMENTUM == 0.9
    assert WEIGHT_DECAY == 0.001
    assert MAX_EPOCHS == 300
    assert TRAIN_BATCH_SIZE == 16
    assert EVAL_BATCH_SIZE == 8


def test_aggregate_probabilities_returns_class_and_log_weighted_concentration():
    result = aggregate_probabilities(np.array([[0, 0, 0, 0, 1, 0, 0]], dtype=float))

    assert result.class_index == 4
    assert result.label == "10^5 CFU/mL"
    assert result.probabilities == (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert result.log10_cfu_ml == 5.0
    assert result.cfu_ml == 100000.0


def test_aggregate_probabilities_rejects_invalid_probability_vectors():
    with np.testing.assert_raises_regex(ValueError, "seven"):
        aggregate_probabilities(np.ones((1, 6)))
    with np.testing.assert_raises_regex(ValueError, "sum"):
        aggregate_probabilities(np.ones((1, 7)))


def test_species_checkpoint_requires_matching_class_order(tmp_path):
    checkpoint = tmp_path / "species.pt"
    import torch

    torch.save({"class_names": ["rongzaohujun", "dachangganjun", "furongxuehujun"]}, checkpoint)

    with np.testing.assert_raises(SpeciesCheckpointError):
        validate_species_checkpoint(checkpoint)


def test_stratified_split_keeps_duplicate_original_images_in_one_split():
    items = [
        ("dachangganjun", "chip-a_0_0.png"), ("dachangganjun", "chip-a_128_0.png"),
        ("dachangganjun", "chip-b.png"), ("dachangganjun", "chip-c.png"),
        ("furongxuehujun", "chip-d.png"), ("furongxuehujun", "chip-e.png"),
        ("furongxuehujun", "chip-f.png"), ("rongzaohujun", "chip-g.png"),
        ("rongzaohujun", "chip-h.png"), ("rongzaohujun", "chip-i.png"),
    ]
    split = make_stratified_split(items, seed=42)

    locations = {name: partition for partition, entries in split.items() for _, name in entries}
    assert locations["chip-a_0_0.png"] == locations["chip-a_128_0.png"]
    assert set(split) == {"train", "test"}


def test_stratified_split_uses_eighty_twenty_ratio_per_species():
    items = [
        (species, f"{species}-{index}.png")
        for species in ("dachangganjun", "furongxuehujun", "rongzaohujun")
        for index in range(10)
    ]

    split = make_stratified_split(items, seed=42)

    for species in ("dachangganjun", "furongxuehujun", "rongzaohujun"):
        assert sum(item_species == species for item_species, _ in split["train"]) == 8
        assert sum(item_species == species for item_species, _ in split["test"]) == 2


def test_write_results_uses_csv_when_requested(tmp_path):
    output = tmp_path / "result.csv"
    write_results(output, [{"image": "a.png", "status": "ok", "class_index": 1}, {"image": "b.png", "status": "error", "error": "bad"}])

    assert output.read_text(encoding="utf-8-sig").splitlines()[0] == "image,status,class_index,error"
