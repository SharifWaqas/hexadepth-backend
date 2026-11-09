"""EXPERIMENTAL small CNN prioritizer. Learns injected synthetic target-like ROIs, not
real sonar classes. It must not be presented as verified object recognition.
No downloads and no labels from real scenes are required. See docs/training-status.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / 'models' / 'roi_ranker_synthetic.pt'


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise RuntimeError('Install PyTorch to use the optional trained ROI model') from e
    return torch, nn


def network():
    torch, nn = _torch()
    return nn.Sequential(
        nn.Conv2d(1, 12, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        nn.Linear(32*4*4, 48), nn.ReLU(), nn.Linear(48, 1),
    )


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize intensity once per image (same transform at training and inference)."""
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (2, 98))
    if hi <= lo + 1:  # almost uniform
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image-lo)/(hi-lo), 0, 1)


def make_background(gray: np.ndarray, size=320) -> np.ndarray:
    h, w = gray.shape
    scale = max(1., size / min(h, w)) if min(h,w)<size else size / max(h,w)
    return cv2.resize(gray, (max(size, int(w*scale)), max(size,int(h*scale))),
                      interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)


def inject_target(patch: np.ndarray, rng: np.random.Generator, novel=False) -> np.ndarray:
    """Create procedural sonar-return/shadow targets for weak synthetic supervision."""
    patch = patch.copy()
    h,w=patch.shape
    mask=np.zeros((h,w),np.uint8)
    cx=int(rng.integers(18, w-18)); cy=int(rng.integers(19, h-18))
    rx=int(rng.integers(7,15)); ry=int(rng.integers(6,13))
    shape=int(rng.integers(3,6)) if novel else int(rng.integers(0,3))
    if shape==0: cv2.ellipse(mask,(cx,cy),(rx,ry),int(rng.integers(0,180)),0,360,255,-1)
    elif shape==1: cv2.rectangle(mask,(cx-rx,cy-ry),(cx+rx,cy+ry),255,-1)
    elif shape==2:
        pts=np.array([[cx-rx,cy+ry],[cx+rx,cy+ry],[cx+int(rng.integers(-3,4)),cy-ry]],np.int32)
        cv2.fillPoly(mask,[pts],255)
    elif shape==3: # elongated unknown object
        cv2.ellipse(mask,(cx,cy),(rx+3,max(2,ry//3)),28,0,360,255,-1)
    elif shape==4: # angular cross
        cv2.rectangle(mask,(cx-3,cy-ry),(cx+3,cy+ry),255,-1)
        cv2.rectangle(mask,(cx-rx,cy-3),(cx+rx,cy+3),255,-1)
    else: # irregular polygon
        theta=np.arange(7)*2*np.pi/7
        radius=rng.uniform(6,17,size=len(theta))
        pts=np.stack([cx+radius*np.cos(theta),cy+radius*np.sin(theta)],axis=1).astype(np.int32)
        cv2.fillPoly(mask,[pts],255)
    dx=int(rng.integers(5,15)); dy=int(rng.integers(1,8))
    shadow=cv2.warpAffine(mask,np.float32([[1,0,dx],[0,1,dy]]),(w,h)) > 0
    arr=patch.astype(np.float32)
    contrast=float(rng.uniform(0.40,0.85))
    arr[shadow] *= float(rng.uniform(.08,.30))
    roi=mask>0
    arr[roi]=np.maximum(arr[roi],255*contrast+arr[roi]*(1-contrast))
    arr=cv2.GaussianBlur(arr,(3,3),float(rng.uniform(.1,.85)))
    return np.clip(arr,0,255).astype(np.uint8)


def sample_patch(backgrounds, rng, positive: bool, novel=False):
    base=backgrounds[int(rng.integers(len(backgrounds)))]
    h,w=base.shape
    size=64
    y=int(rng.integers(0,h-size+1));x=int(rng.integers(0,w-size+1))
    patch=base[y:y+size,x:x+size].copy()
    if positive:
        patch=inject_target(patch,rng,novel=novel)
    if rng.random()<.5: patch=cv2.flip(patch, 1)
    if rng.random()<.5: patch=cv2.flip(patch, 0)
    # Sonar-specific gain shift and weak speckle/noise, independent on each sample.
    arr=patch.astype(np.float32)
    arr=arr*float(rng.uniform(.78,1.2))+float(rng.uniform(-9,9))
    arr += rng.normal(0,rng.uniform(1,8),size=arr.shape)
    patch=np.clip(arr,0,255).astype(np.uint8)
    return normalize(patch)


class PriorityRanker:
    def __init__(self, checkpoint: Path | None = None):
        torch,_=_torch()
        self.device='cpu'
        torch.set_num_threads(min(3,torch.get_num_threads()))
        self.net=network()
        p=Path(checkpoint or os.environ.get('HEXADEPTH_ROI_WEIGHTS',WEIGHTS))
        if not p.is_file():
            raise FileNotFoundError(p)
        state=torch.load(p,map_location='cpu',weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval()

    def detect(self, gray: np.ndarray, count=2):
        torch,_=_torch()
        if gray.ndim != 2 or min(gray.shape) < 32:
            return []
        h,w=gray.shape
        # Fixed-size normalized view; each 64px receptive field covers ~1/5 of scene.
        view=cv2.resize(gray,(320,320),interpolation=cv2.INTER_AREA if max(gray.shape)>=320 else cv2.INTER_LINEAR)
        v=normalize(view)
        stride=24
        xy=[(x,y) for y in range(0,257,stride) for x in range(0,257,stride)]
        if (256,256) not in xy: xy.append((256,256))
        patches=np.stack([v[y:y+64,x:x+64] for x,y in xy]).astype('float32')
        with torch.no_grad():
            pred=self.net(torch.from_numpy(patches[:,None])).flatten().sigmoid().numpy()
        order=np.argsort(-pred)
        out=[]
        for idx in order:
            x,y=xy[int(idx)];p=float(pred[idx]);
            # abstain if evidence is weak; caller falls back to saliency for the overview
            if p < .65: break
            box=[x/320*w,y/320*h,(x+64)/320*w,(y+64)/320*h]
            if any(max(0,min(box[2],q[2])-max(box[0],q[0]))*max(0,min(box[3],q[3])-max(box[1],q[1])) > .22*(box[2]-box[0])*(box[3]-box[1]) for q in [r['box_px'] for r in out]):
                continue
            out.append(dict(box_px=box,class_id=255,confidence=int(round(p*100))))
            if len(out)>=count:break
        return out
