"""CPU-only experimental training: synthetic target injections, multiple *image sources*.
NOT labeled multi-dataset detection. Reports held-out novel-shape performance separately.
Requires: torch, numpy, opencv-python, scikit-learn.

python scripts/train_priority_ranker.py --epochs 6 --train-count 2200 --val-count 450
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np, cv2, torch
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,confusion_matrix
from hexadepth.learned_roi import make_background,sample_patch,network,WEIGHTS


def collect():
    real=sorted((ROOT/'web/samples/real').glob('*.jpg'))
    synth=sorted((ROOT/'samples').glob('*.png'))
    assert len(real)>=4 and len(synth)>=5, (len(real),len(synth))
    # Leave the entire 2021 real source and two synthetic scenes unseen during fitting.
    train_real=[p for p in real if '2021' not in p.name]
    test_real=[p for p in real if '2021' in p.name]
    assert test_real
    train_synth=synth[:3]
    test_synth=synth[3:]
    def load(ps):
        return [make_background(cv2.imread(str(p),0)) for p in ps]
    return (load(train_real+train_synth), load(test_real+test_synth),
            {'train_real_files':[p.name for p in train_real],
             'train_synthetic_files':[p.name for p in train_synth],
             'heldout_real_files':[p.name for p in test_real],
             'heldout_synthetic_files':[p.name for p in test_synth]})


def dataset(backgrounds,n,seed,novel=False):
    rng=np.random.default_rng(seed)
    labels=np.arange(n)%2
    rng.shuffle(labels)
    x=np.stack([sample_patch(backgrounds,rng,bool(z),novel=novel) for z in labels]).astype('float32')
    return torch.from_numpy(x[:,None]),torch.from_numpy(labels.astype('float32'))


def metrics(y,p,cutoff=.65):
    y=np.asarray(y,dtype=int);p=np.asarray(p)
    pred=p>=cutoff
    prec,rec,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),roc_auc=float(roc_auc_score(y,p)),precision=float(prec),recall=float(rec),
                f1=float(f1),false_positive_rate=float(fp/max(1,fp+tn)),
                true_positive=int(tp),false_positive=int(fp),false_negative=int(fn),true_negative=int(tn),
                threshold=cutoff)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs',type=int,default=6)
    ap.add_argument('--train-count',type=int,default=2200)
    ap.add_argument('--val-count',type=int,default=450)
    ap.add_argument('--batch-size',type=int,default=96)
    a=ap.parse_args()
    torch.manual_seed(21);torch.set_num_threads(3)
    train_bg,test_bg,sources=collect()
    # Heldout data is constructed ONCE, never used for fitting or hyperparameter selection.
    X,y=dataset(train_bg,a.train_count,900)
    Xv,yv=dataset(test_bg,a.val_count,901,novel=False)
    Xnov,ynov=dataset(test_bg,a.val_count,902,novel=True)
    model=network(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    criterion=torch.nn.BCEWithLogitsLoss()
    logs=[]; start=time.perf_counter()
    for ep in range(a.epochs):
        model.train();perm=torch.randperm(len(y));total=0
        for ids in perm.split(a.batch_size):
            pred=model(X[ids]).flatten()
            loss=criterion(pred,y[ids]);opt.zero_grad();loss.backward();opt.step()
            total+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            probs=model(Xv).flatten().sigmoid().numpy()
        log={'epoch':ep+1,'training_loss':total/len(y), 'heldout_auc_monitor':float(roc_auc_score(yv.numpy(),probs))}
        logs.append(log)
        print('epoch',ep+1,'train_loss',round(log['training_loss'],4),'heldout_AUC',round(log['heldout_auc_monitor'],4),flush=True)
    # The holdout AUC is reported each epoch for transparency; early stopping not used.
    model.eval()
    with torch.no_grad():
        p=model(Xv).flatten().sigmoid().numpy()
        pn=model(Xnov).flatten().sigmoid().numpy()
    W=ROOT/'models';W.mkdir(exist_ok=True)
    torch.save(model.state_dict(),WEIGHTS)
    report={'model':'small CNN synthetic-return ROI ranker','objective':'injected target-like pattern vs background',
            'real_object_detection_validated':False,'has_semantic_classes':False,
            'trained_on_labeled_external_datasets':False,'sources':sources,
            'train_patches':len(y),'heldout_patches':len(yv),'novel_shape_patches':len(ynov),
            'heldout_same_shapes':metrics(yv.numpy(),p),'heldout_novel_shapes':metrics(ynov.numpy(),pn),
            'epochs':a.epochs,'epoch_logs':logs,'elapsed_s':round(time.perf_counter()-start,2),
            'weights_bytes':WEIGHTS.stat().st_size, 'seed':21,'input_patch':64,
            'important_limit':'All positive targets are synthetic insertions; test does NOT validate real target recognition; patch-level AUC may overestimate scene detection quality.'}
    out=ROOT/'results'/'roi_training_metrics.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print('model:',WEIGHTS,'metrics:',out)
    print('heldout shape AUC:',report['heldout_same_shapes']['roc_auc'], 'novel-shape AUC:',report['heldout_novel_shapes']['roc_auc'])
if __name__=='__main__':main()
