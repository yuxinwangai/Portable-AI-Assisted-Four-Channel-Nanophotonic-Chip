import numpy as np

class NoForegroundError(ValueError):
    pass

def segment(rgb, cfg):
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, coarse = cv2.threshold(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(cv2.bitwise_not(coarse), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise NoForegroundError('No foreground contour')
    mask = np.zeros_like(coarse)
    cv2.fillPoly(mask, contours, 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise NoForegroundError('No foreground component')
    label = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    mask = labels == label
    yy, xx = np.where(mask)
    crop = (bgr * mask[..., None])[yy.min():yy.max()+1, xx.min():xx.max()+1]
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, fine = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fine = cv2.medianBlur(fine, cfg['segmentation_median_kernel'])
    yy, xx = np.where(fine == 255)
    if not len(xx):
        raise NoForegroundError('No fine foreground region')
    return cv2.cvtColor(crop[yy.min():yy.max()+1, xx.min():xx.max()+1], cv2.COLOR_BGR2RGB)

def extract(rgb, cfg):
    import cv2
    size, stride = cfg['patch_size'], cfg['stride']
    if size <= 0 or stride <= 0:
        raise ValueError('Patch size and stride must be positive')
    patches, locations = [], []
    h, w = rgb.shape[:2]
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = rgb[y:min(y+size,h), x:min(x+size,w)]
            ph, pw = patch.shape[:2]
            if 1 - ph*pw/size**2 > cfg['max_missing_fraction']:
                continue
            patch = np.pad(patch, ((0,size-ph),(0,size-pw),(0,0)), constant_values=0)
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            if np.mean(gray == 0) > cfg['max_black_fraction']:
                continue
            _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            binary = cv2.medianBlur(binary,cfg['patch_median_kernel'])
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if n <= 1:
                continue
            label = 1 + int(stats[1:,cv2.CC_STAT_AREA].argmax())
            area = int(stats[label,cv2.CC_STAT_AREA])
            contours, _ = cv2.findContours((labels==label).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            _, (rw,rh), _ = cv2.minAreaRect(contours[0])
            short = min(rw,rh)
            if short and max(rw,rh)/short > cfg['min_aspect_ratio'] and cfg['min_component_area'] < area < cfg['max_component_area']:
                patches.append(patch)
                locations.append({'x':x,'y':y,'width':pw,'height':ph})
    return patches, locations

def prepare(path, mode, cfg):
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    if mode == 'patch':
        return [rgb], [{'x':0,'y':0,'width':rgb.shape[1],'height':rgb.shape[0]}]
    if mode not in ('raw','segmented'):
        raise ValueError('Input mode must be raw, segmented or patch')
    if mode == 'raw':
        try:
            rgb = segment(rgb,cfg)
        except NoForegroundError:
            return [], []
    return extract(rgb,cfg)
