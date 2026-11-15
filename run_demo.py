
import argparse
import json
from pathlib import Path
from cej_inference.pipeline import predict_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('demo_predictions.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'config/inference.json').read_text())
    rows = json.loads((root / 'demo/image_manifest.json').read_text())
    output = []
    for row in rows:
        print(row['sample_id'], flush=True)
        result = predict_image(root / row['file'], row['input_mode'], config, root, args.device)
        result['input'] = row['file']
        output.append({'sample_id': row['sample_id'], **result})
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(args.output)

if __name__ == '__main__':
    main()
