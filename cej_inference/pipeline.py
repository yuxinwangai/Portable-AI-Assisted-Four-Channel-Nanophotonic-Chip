import numpy as np
from .core import SPECIES, probabilities, report

def evaluate(patches, infer, mapping, target=None):
    if target is not None and target not in SPECIES:raise ValueError('Unknown target species')
    if not patches:
        return {'status':'non_evaluable','valid_patch_count':0,'image_species':None,'targets':{}}
    species_scores=probabilities(infer('species',patches),3)
    if len(species_scores)!=len(patches):raise ValueError('Species rows must match patch count')
    assignments=species_scores.argmax(axis=1)
    burden={}
    for index,name in enumerate(SPECIES):
        group=[p for p,a in zip(patches,assignments) if a==index]
        if group:burden[name]=infer(name,group)
    return report(species_scores,burden,mapping,target)

def predict_image(path,mode,config,root,device='cpu',target=None):
    from .preprocess import prepare
    from .models import predict
    patches,locations=prepare(path,mode,config['lfpe'])
    def infer(task,images):
        return predict(images,config['models'][task],root,device,config['batch_size'])
    out=evaluate(patches,infer,config['burden_mapping'],target)
    out['input']=str(path);out['input_mode']=mode;out['patch_locations']=locations
    return out
