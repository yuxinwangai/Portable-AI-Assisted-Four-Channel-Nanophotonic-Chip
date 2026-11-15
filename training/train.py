import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['Escherichia coli', 'Vibrio parahaemolyticus', 'Vibrio alginolyticus']
TASKS = ['species', 'dachangganjun', 'furongxuehujun', 'rongzaohujun']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import resnet50
    from species_model import ResNet50

    cfg = json.loads((ROOT / 'training/config.json').read_text())[args.task]
    spec = json.loads((ROOT / 'config/inference.json').read_text())['models'][args.task]
    manifest = args.manifest or ROOT / 'splits' / ('Species patch split.csv' if args.task == 'species' else 'Burden patch split.csv')
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.task != 'species':
        rows = [r for r in rows if r['Species'] == NAMES[TASKS.index(args.task)-1]]
    norm = spec['normalization']
    if args.task == 'species':
        norm = norm['legacy_resnet50']

    class Patches(Dataset):
        def __init__(self, subset):
            self.rows = [r for r in rows if r['Subset'] == subset]
            self.training = subset == 'train'
            if not self.rows:
                raise ValueError(f'Empty {subset} partition')
            for row in self.rows:
                path = args.data_root / row['Source file']
                if not row['Source file'] or not path.is_file():
                    raise FileNotFoundError(f"Missing patch for {row['Patch ID']}: {path}")

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(args.data_root / row['Source file']) as image:
                image = image.convert('RGB').resize((spec['input_size'],)*2, Image.Resampling.BILINEAR)
                if self.training and torch.rand(()) < cfg['horizontal_flip']:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if self.training and torch.rand(()) < cfg['vertical_flip']:
                    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                array = np.asarray(image, dtype=np.float32) / norm['pixel_scale']
            array = (array - np.asarray(norm['mean'], dtype=np.float32)) / np.asarray(norm['std'], dtype=np.float32)
            label = NAMES.index(row['Species']) if args.task == 'species' else int(row['Model label'])
            return torch.from_numpy(array.transpose(2, 0, 1).copy()), label

    train = DataLoader(Patches('train'), batch_size=cfg['train_batch_size'], shuffle=True)
    test = DataLoader(Patches('test'), batch_size=cfg['eval_batch_size'], shuffle=False)
    model = ResNet50(3) if args.task == 'species' else resnet50(weights=None, num_classes=7)
    model.to(args.device)
    if cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['T_max']) if 'T_max' in cfg else None
    epochs = args.epochs if args.epochs is not None else cfg['epochs']
    if epochs < 1:
        raise ValueError('epochs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = count = 0
        for images, labels in train:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model(images)
            if args.task == 'species':
                logits = logits[0]
                loss = F.cross_entropy(logits, labels)
            else:
                targets = F.one_hot(labels, 7).to(logits.dtype)
                probability = logits.sigmoid()
                pt = (1-probability)*targets + probability*(1-targets)
                weight = (cfg['alpha']*targets + (1-cfg['alpha'])*(1-targets))*pt.pow(cfg['gamma'])
                loss = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none')*weight).sum()/len(labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(labels)
            count += len(labels)
        if scheduler:
            scheduler.step()
        state = model.state_dict()
        if args.task != 'species':
            state = {('head.fc.'+k[3:] if k.startswith('fc.') else 'backbone.'+k): v for k,v in state.items()}
        torch.save({k:v.detach().cpu() for k,v in state.items()}, args.output / f'epoch_{epoch+1}.pth')
        print(json.dumps({'epoch':epoch+1, 'loss':total_loss/count}), flush=True)
    model.eval()
    correct = count = 0
    with torch.no_grad():
        for images, labels in test:
            logits = model(images.to(args.device))
            if args.task == 'species':
                logits = logits[0]
            correct += int((logits.argmax(1).cpu()==labels).sum())
            count += len(labels)
    (args.output/'evaluation.json').write_text(json.dumps({'correct':correct,'total':count,'accuracy':correct/count},indent=2))

if __name__ == '__main__':
    main()
