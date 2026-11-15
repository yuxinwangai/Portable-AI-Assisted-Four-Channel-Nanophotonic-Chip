from collections import Counter
import numpy as np

SPECIES = ('dachangganjun', 'furongxuehujun', 'rongzaohujun')

def probabilities(values, classes):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != classes or not len(p):
        raise ValueError(f'Expected a nonempty N x {classes} probability matrix')
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Probabilities must be finite and within [0, 1]')
    if not np.allclose(p.sum(axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError('Each probability row must sum to one')
    return p / p.sum(axis=1, keepdims=True)

def summarize_burden(values, mapping):
    p = probabilities(values, 7)
    offset = float(mapping['log10_offset'])
    step = float(mapping['log10_step'])
    anchors = offset + step * np.arange(7, dtype=np.float64)
    if anchors.shape != (7,) or not np.all(np.isfinite(anchors)):
        raise ValueError('Expected seven finite log10 mapping anchors')
    if np.any(np.diff(anchors) <= 0) or anchors[0] < -300 or anchors[-1] > 300:
        raise ValueError('Mapping anchors must increase within the supported numeric range')

    exponent = p @ anchors
    if np.any(exponent > 300):
        raise ValueError('Mapping anchors cause an unsupported numeric range')
    scores = np.power(10., exponent)
    average = p.mean(axis=0)
    return {'grade': int(average.argmax()) + 1,
            'grade_probabilities': average.tolist(),
            'continuous_output': float(scores.mean()),
            'patch_count': len(p)}

def report(species_scores, burden_scores, mapping, target=None):
    if target is not None and target not in SPECIES:
        raise ValueError('Unknown target species')
    p = probabilities(species_scores, 3)
    labels = p.argmax(axis=1)
    groups = {}
    for index, name in enumerate(SPECIES):
        count = int(np.sum(labels == index))
        if count:
            q = probabilities(burden_scores[name], 7)
            if len(q) != count:
                raise ValueError(f'{name}: burden rows must match assigned patches')
            groups[name] = {'detected': True, **summarize_burden(q, mapping)}
        else:
            groups[name] = {'detected': False, 'grade': None,
                            'continuous_output': None, 'patch_count': 0}
    mean = p.mean(axis=0)
    return {'status': 'ok', 'valid_patch_count': len(p),
            'image_species': SPECIES[int(mean.argmax())],
            'image_species_probabilities': mean.tolist(),
            'targets': groups if target is None else {target: groups[target]}}

def majority_vote(values):
    if len(values) != 3:
        raise ValueError('Chip aggregation requires exactly three image labels')
    counts = Counter(v for v in values if v is not None)
    winners = [v for v, n in counts.items() if n >= 2]
    return winners[0] if winners else None
