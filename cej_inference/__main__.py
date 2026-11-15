import argparse,json,sys
from pathlib import Path
from .core import SPECIES,report,majority_vote

ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
    parser=argparse.ArgumentParser(prog='python -m cej_inference')
    parser.add_argument('--config',type=Path,default=ROOT/'config/inference.json')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['predict','preprocess']:
        p=sub.add_parser(name);p.add_argument('input',type=Path);p.add_argument('--mode',choices=['raw','segmented','patch'],default='raw');p.add_argument('--output',type=Path,required=True)
        if name=='predict':p.add_argument('--device',default='cpu');p.add_argument('--target',choices=SPECIES)
    p=sub.add_parser('from-probabilities');p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--target',choices=SPECIES)
    p=sub.add_parser('aggregate-chip');p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    sub.add_parser('check')
    args=parser.parse_args(argv)
    try:
        config=json.loads(args.config.read_text(encoding='utf-8-sig'))
        if tuple(config['species_order']) != SPECIES or config['grade_order'] != list(range(1,8)):
            raise ValueError('Configured class order does not match this inference implementation')
        if not isinstance(config['batch_size'],int) or config['batch_size'] < 1:
            raise ValueError('batch_size must be a positive integer')
        if args.command=='check':
            records={k:{'path':str(ROOT/v['checkpoint']),'present':(ROOT/v['checkpoint']).is_file(),'classes':v['classes']} for k,v in config['models'].items()}
            print(json.dumps(records,ensure_ascii=False,indent=2));return 0 if all(r['present'] for r in records.values()) else 2
        if args.command=='predict':
            from .pipeline import predict_image
            result=predict_image(args.input,args.mode,config,ROOT,args.device,args.target)
        elif args.command=='preprocess':
            from .preprocess import prepare
            patches,locations=prepare(args.input,args.mode,config['lfpe']);result={'status':'ok' if patches else 'non_evaluable','valid_patch_count':len(patches),'patch_locations':locations}
        elif args.command=='from-probabilities':
            data=json.loads(args.input.read_text(encoding='utf-8-sig'));result=report(data['species_probabilities'],data['burden_probabilities'],config['burden_mapping'],args.target)
        else:
            data=json.loads(args.input.read_text(encoding='utf-8-sig'));species=majority_vote(data['species_labels']);grade=majority_vote(data['grade_labels']);result={'species':species,'grade':grade,'status':'determinate' if species is not None and grade is not None else 'indeterminate','image_count':3}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(args.output);return 0
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
