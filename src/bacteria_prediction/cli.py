import argparse
import csv
import json
from pathlib import Path
from importlib.metadata import distribution
import subprocess
import sys

from .concentration import predict_concentration
from .data import prepare_species_dataset
from .species import predict_species
from .training import MAX_EPOCHS, TRAIN_BATCH_SIZE, evaluate_species, train_species


def _images(path: Path):
    return [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif"})


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fields = ["image", "status"]
        for row in rows:
            fields.extend(key for key in row if key not in fields)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bacteria-predict")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("predict-concentration", "predict-species"):
        command = commands.add_parser(name); command.add_argument("input", type=Path); command.add_argument("--output", type=Path, default=Path("prediction_results.json")); command.add_argument("--device", default="cpu")
        if name == "predict-concentration": command.add_argument("--species", required=True); command.add_argument("--input-mode", default="raw", choices=("raw", "segmented", "patch"))
        else: command.add_argument("--checkpoint", type=Path)
    prepare = commands.add_parser("prepare-species-data"); prepare.add_argument("source", type=Path); prepare.add_argument("destination", type=Path); prepare.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train"); train.add_argument("task", choices=("species", "concentration")); train.add_argument("dataset", type=Path); train.add_argument("--species", choices=("dachangganjun", "furongxuehujun", "rongzaohujun")); train.add_argument("--output", type=Path, default=Path("models/species/best_species.pt")); train.add_argument("--epochs", type=int, default=MAX_EPOCHS); train.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE); train.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("dataset", type=Path); evaluate.add_argument("checkpoint", type=Path); evaluate.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "prepare-species-data": print(json.dumps(prepare_species_dataset(args.source, args.destination, args.seed), ensure_ascii=False)); return 0
    if args.command == "train":
        if args.task == "concentration":
            if not args.species: raise SystemExit("Concentration training requires --species")
            from .registry import get_species, package_root
            root = package_root(); model = get_species(args.species); config = root / "configs" / "concentration" / model.config_file
            train_script = Path(distribution("mmpretrain").locate_file("mmpretrain/.mim/tools/train.py"))
            if not train_script.is_file():
                raise FileNotFoundError(f"Training entry point not found: {train_script}")
            command = [sys.executable, str(train_script), str(config), "--work-dir", str(args.output), "--cfg-options", f"train_dataloader.dataset.data_root={args.dataset}", f"val_dataloader.dataset.data_root={args.dataset}", f"test_dataloader.dataset.data_root={args.dataset}"]
            subprocess.run(command, check=True)
            return 0
        print(train_species(args.dataset, args.output, args.epochs, args.batch_size, args.device)); return 0
    if args.command == "evaluate": print(json.dumps(evaluate_species(args.dataset, args.checkpoint, args.device), ensure_ascii=False)); return 0
    rows = []
    for image in _images(args.input):
        try:
            result = predict_concentration(image, args.species, args.input_mode, args.device).to_dict() if args.command == "predict-concentration" else predict_species(image, args.checkpoint, args.device).to_dict()
            rows.append({"image": str(image), "status": "ok", **result})
        except Exception as error: rows.append({"image": str(image), "status": "error", "error": f"{type(error).__name__}: {error}"})
    write_results(args.output, rows)
    print(args.output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
