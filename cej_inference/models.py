
from pathlib import Path
import gc
import re
import pickle
import numpy as np

def load(path, classes, device):
    import torch
    from torchvision.models import resnet50
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError('This checkpoint is not a plain state dictionary. Export model.state_dict() in its original model environment before use.') from error
    if not isinstance(checkpoint,dict):
        raise ValueError('Checkpoint must contain a state dictionary')
    state = checkpoint.get('state_dict',checkpoint.get('model_state_dict',checkpoint))
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('Unsupported checkpoint structure')
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    if 'conv1.0.weight' in state:
        architecture = 'legacy_resnet50'
    elif 'backbone.conv1.weight' in state:
        architecture = 'mmpretrain_resnet50'
    elif 'conv1.weight' in state and 'fc.0.weight' in state:
        architecture = 'torchvision_resnet50'
    else:
        raise ValueError('Unsupported ResNet state keys; no partial loading is performed')
    network = resnet50(weights=None)
    if architecture == 'mmpretrain_resnet50' and 'head.fc.weight' in state:
        network.fc = torch.nn.Linear(2048,classes)
    elif architecture == 'legacy_resnet50':
        network.fc = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.Linear(1024,classes))
    else:
        network.fc = torch.nn.Sequential(torch.nn.Linear(2048,1024),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(1024,classes))
    converted = {}
    block = {'0':'conv1','1':'bn1','3':'conv2','4':'bn2','6':'conv3','7':'bn3'}
    for key,value in state.items():
        if architecture == 'legacy_resnet50':
            if key.startswith('conv1.0.'): key='conv1.'+key[len('conv1.0.'):]
            elif key.startswith('conv1.1.'): key='bn1.'+key[len('conv1.1.'):]
            elif key.startswith('fc1.'): key='fc.1.'+key[4:]
            elif key.startswith('fc.'): key='fc.0.'+key[3:]
            else:
                key=re.sub(r'(layer\d+\.\d+)\.bottleneck\.(0|1|3|4|6|7)\.',lambda m:m[1]+'.'+block[m[2]]+'.',key)
        elif architecture == 'mmpretrain_resnet50':
            if key.startswith('backbone.'): key=key[len('backbone.'):]
            elif key.startswith('head.fc.'): key='fc.'+key[len('head.fc.'):]
            elif key.startswith('head.layers.0.fc.'): key='fc.0.'+key[len('head.layers.0.fc.'):]
            elif key.startswith('head.layers.1.fc.'): key='fc.3.'+key[len('head.layers.1.fc.'):]
            elif key in ('data_preprocessor.mean','data_preprocessor.std'): continue
        if key in converted: raise ValueError(f'Duplicate mapped state key: {key}')
        converted[key]=value
    network.load_state_dict(converted,strict=True)
    del state,converted,checkpoint
    return network.to(device).eval(), architecture

def predict(patches, spec, root, device='cpu', batch_size=1):
    import torch
    from PIL import Image
    path=(Path(root)/spec['checkpoint']).resolve()
    if not path.is_file(): raise FileNotFoundError(f'Missing checkpoint: {path}')
    torch.set_num_threads(1)
    network,architecture=load(path,spec['classes'],device)
    norm=spec['normalization']
    if 'mean' not in norm:
        if architecture not in norm: raise ValueError(f'No normalization configured for {architecture}')
        norm=norm[architecture]
    mean=np.asarray(norm['mean'],dtype=np.float32);std=np.asarray(norm['std'],dtype=np.float32)
    if np.any(std<=0): raise ValueError('Normalization std must be positive')
    outputs=[]
    try:
        with torch.inference_mode():
            for start in range(0,len(patches),batch_size):
                arrays=[]
                for rgb in patches[start:start+batch_size]:
                    size=spec['input_size'];im=Image.fromarray(rgb)
                    if im.size!=(size,size):im=im.resize((size,size),Image.Resampling.BILINEAR)
                    x=np.asarray(im,dtype=np.float32)/norm['pixel_scale'];x=(x-mean)/std
                    arrays.append(torch.from_numpy(x.transpose(2,0,1).copy()))
                tensor=torch.stack(arrays).to(device)
                outputs.extend(torch.softmax(network(tensor),dim=1).cpu().numpy().tolist())
    finally:
        del network
        gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return outputs
