Download the four `.pth` files from [v1.0.0](https://github.com/yuxinwangai/Portable-AI-Assisted-Four-Channel-Nanophotonic-Chip/releases/tag/v1.0.0) and place them in `weights/` before running inference.

Use Python 3.10.12. From this directory:

```shell
python -m pip install -r requirements.txt
python run_demo.py --device cpu
```

The demo processes the six supplied images and saves results to `demo_predictions.json`.

For another image:

```shell
python -m cej_inference predict input.png --mode raw --output result.json
```

Use `--mode raw` for an original image or `--mode segmented` for a foreground-cropped image. Add `--device cuda` to use a compatible CUDA-enabled PyTorch installation.

Keep the supplied `weights/` and `config/` directories alongside the code.
