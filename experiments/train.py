"""Запуск: python -m experiments.train --help из корня ЛР 3."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.preprocessing import preprocess

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    model: str = 'resnet18.a1_in1k'
    seed: int = 42
    image_size: int = 160
    batch_size: int = 8
    epochs: int = 6
    freeze_epochs: int = 2
    lr_head: float = 0.001
    lr_backbone: float = 0.0001
    weight_decay: float = 0.0001
    num_workers: int = 0
    threads: int = 2
    run_name: str = 'resnet18_lr001'
    data_dir: str = 'experiments/data/raw'
    manifest: str = 'experiments/data/processed/split.csv'
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def prepare_split(config):
    """Fixed author-disjoint split, approximately 60/20/20 per class."""
    root = ROOT/config.data_dir
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) < 3:
        raise ValueError('Нужны минимум 3 папки классов в data/raw.')
    rows, seen = [], set()
    for label in classes:
        files = sorted(p for p in (root/label).iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp'])
        if len(files) < 30:raise ValueError(f'{label}: требуется минимум 30 изображений')
        random.Random(config.seed).shuffle(files)
        n = len(files); ntrain = int(n*.6); nval = int(n*.2)
        for i,p in enumerate(files):
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                digest = hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest()
            if digest in seen:raise ValueError(f'Повторное изображение: {p}')
            seen.add(digest)
            split = 'train' if i<ntrain else 'val' if i<ntrain+nval else 'test'
            rows.append(dict(path=p.relative_to(ROOT).as_posix(),label=label,split=split,sha256=digest))
    # All photographs from the same credited author stay in a single split.
    # This prevents nearby photographs from one shooting session leaking across splits.
    sources=ROOT/'experiments/data/sources.csv'
    if sources.exists():
        with sources.open(encoding='utf-8') as f:provenance={r['path']:r for r in csv.DictReader(f)}
        for r in rows:r['group']=provenance[r['path']]['author'] or provenance[r['path']]['source_page']
        groups=sorted({r['group'] for r in rows});rng=np.random.default_rng(config.seed)
        counts=np.array([[sum(r['group']==g and r['label']==c for r in rows) for c in classes] for g in groups])
        totals=counts.sum(0);best_score=float('inf');assignment=None
        for _ in range(20000):
            labels=rng.choice(3,len(groups),p=[.6,.2,.2])
            sizes=np.stack([counts[labels==s].sum(0) for s in range(3)])
            if (sizes[0]<15).any() or (sizes[1:]<5).any():continue
            score=np.square(sizes/totals-np.array([.6,.2,.2])[:,None]).sum()
            if score<best_score:best_score=score;assignment=labels.copy()
        if assignment is None:raise ValueError('Not enough independent authors for a balanced split')
        mapping={g:['train','val','test'][s] for g,s in zip(groups,assignment)}
        for r in rows:r['split']=mapping[r['group']]
    dest=ROOT/config.manifest;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


class Images(Dataset):
    def __init__(self, rows, split, classes, config):
        self.rows=[r for r in rows if r['split']==split]
        self.classes=classes;self.config=asdict(config);self.training=split=='train'
        self.augment=T.Compose([T.RandomResizedCrop(config.image_size,scale=(.75,1.0)),
                      T.RandomHorizontalFlip(),T.ColorJitter(.15,.15,.1,.03)])
    def __len__(self):return len(self.rows)
    def __getitem__(self, index):
        row=self.rows[index]
        with Image.open(ROOT/row['path']) as im:
            image=ImageOps.exif_transpose(im).convert('RGB')
            if self.training:image=self.augment(image)
            x=torch.from_numpy(preprocess(image,self.config))
        return x,self.classes.index(row['label'])


def evaluate(model, loader):
    model.eval(); ys=[]; preds=[]; total=0.0
    with torch.no_grad():
        for x,y in loader:
            z=model(x); total+=nn.functional.cross_entropy(z,y,reduction='sum').item()
            ys.extend(y.tolist());preds.extend(z.argmax(1).tolist())
    return total/len(ys),float(accuracy_score(ys,preds)),ys,preds


def train(config):
    set_seed(config.seed);torch.set_num_threads(config.threads)
    manifest=ROOT/config.manifest
    if not manifest.exists():prepare_split(config)
    with manifest.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    classes=sorted({r['label'] for r in rows})
    model=timm.create_model(config.model,pretrained=True,num_classes=len(classes))
    config.mean=tuple(model.pretrained_cfg['mean']);config.std=tuple(model.pretrained_cfg['std'])
    loaders={s:DataLoader(Images(rows,s,classes,config),batch_size=config.batch_size,shuffle=s=='train',
                  num_workers=config.num_workers,generator=torch.Generator().manual_seed(config.seed)) for s in ['train','val']}
    head=list(model.get_classifier().parameters());head_ids={id(p) for p in head}
    backbone=[p for p in model.parameters() if id(p) not in head_ids]
    history=[];best=(-1,float('-inf'));out=ROOT/'experiments/models';out.mkdir(exist_ok=True)
    start=time.perf_counter()
    for epoch in range(config.epochs):
        frozen=epoch<config.freeze_epochs
        if epoch in [0,config.freeze_epochs]:
            for p in backbone:p.requires_grad=not frozen
            optimizer=torch.optim.AdamW([{'params':head,'lr':config.lr_head},
                           {'params':backbone,'lr':0 if frozen else config.lr_backbone}],weight_decay=config.weight_decay)
        # Frozen backbone (including BatchNorm running stats) must remain in eval mode.
        model.train()
        if frozen:
            model.eval();model.get_classifier().train()
        total=correct=count=0
        for x,y in loaders['train']:
            optimizer.zero_grad();z=model(x);loss=nn.functional.cross_entropy(z,y)
            loss.backward();optimizer.step()
            total+=loss.item()*len(y);correct+=(z.argmax(1)==y).sum().item();count+=len(y)
        vl,va,_,_=evaluate(model,loaders['val'])
        row=dict(epoch=epoch+1,stage='head' if frozen else 'all',train_loss=total/count,
                 train_accuracy=correct/count,val_loss=vl,val_accuracy=va)
        history.append(row);print(config.run_name,row,flush=True)
        key=(va,-vl)
        if key>best:
            best=key
            torch.save(dict(state_dict=model.state_dict(),config=asdict(config),classes=classes,
                            epoch=epoch+1,validation_accuracy=va,validation_loss=vl),out/(config.run_name+'.pt'))
    result=dict(config=asdict(config),history=history,seconds=time.perf_counter()-start,
                best_validation_accuracy=best[0],best_validation_loss=-best[1],parameters=sum(p.numel() for p in model.parameters()))
    reports=ROOT/'experiments/reports';reports.mkdir(exist_ok=True)
    (reports/(config.run_name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for ax,metric in zip(axs,['loss','accuracy']):
        for split in ['train','val']:ax.plot([r['epoch'] for r in history],[r[split+'_'+metric] for r in history],label=split)
        ax.axvline(config.freeze_epochs+.5,ls=':',color='gray');ax.set(xlabel='Epoch',ylabel=metric);ax.legend();ax.grid(alpha=.2)
    fig.suptitle(config.run_name);fig.tight_layout();(reports/'figures').mkdir(exist_ok=True)
    fig.savefig(reports/'figures'/(config.run_name+'.png'),dpi=160);plt.close(fig)
    return result


def test_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    config=TrainConfig(**saved['config']);classes=saved['classes'];torch.set_num_threads(config.threads)
    model=timm.create_model(config.model,pretrained=False,num_classes=len(classes))
    model.load_state_dict(saved['state_dict'])
    with (ROOT/config.manifest).open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    loader=DataLoader(Images(rows,'test',classes,config),batch_size=config.batch_size)
    loss,acc,y,p=evaluate(model,loader)
    cm=confusion_matrix(y,p,labels=list(range(len(classes))))
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
    for i in range(len(classes)):
        for j in range(len(classes)):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set(xticks=range(len(classes)),yticks=range(len(classes)),xticklabels=classes,yticklabels=classes,xlabel='Predicted',ylabel='True',title=config.model)
    fig.tight_layout();fig.savefig(ROOT/'experiments/reports/figures'/(config.run_name+'_confusion.png'),dpi=160);plt.close(fig)
    result=dict(accuracy=acc,loss=loss,confusion_matrix=cm.tolist(),classes=classes,predictions=p,targets=y)
    (ROOT/'experiments/reports'/(config.run_name+'_test.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def export_checkpoint(path):
    import onnx
    import onnxruntime as ort
    saved=torch.load(path,map_location='cpu',weights_only=False);config=TrainConfig(**saved['config'])
    set_seed(config.seed);torch.set_num_threads(config.threads)
    model=timm.create_model(config.model,pretrained=False,num_classes=len(saved['classes']))
    model.load_state_dict(saved['state_dict']);model.eval()
    output=ROOT/'app/model.onnx'
    x=torch.randn(1,3,config.image_size,config.image_size)
    torch.onnx.export(model,x,str(output),opset_version=17,input_names=['image'],output_names=['logits'],
                      dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
    onnx.checker.check_model(onnx.load(str(output)))
    session=ort.InferenceSession(str(output),providers=['CPUExecutionProvider'])
    with torch.no_grad():expected=model(x).numpy()
    actual=session.run(None,{'image':x.numpy()})[0]
    np.testing.assert_allclose(expected,actual,atol=1e-4,rtol=1e-3)
    metadata=dict(config=asdict(config),classes=saved['classes'],epoch=saved['epoch'],
                  max_export_error=float(np.max(abs(expected-actual))))
    (ROOT/'app/model.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (ROOT/'experiments/best_config.json').write_text(json.dumps(asdict(config),indent=2),encoding='utf-8')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--test',type=Path)
    parser.add_argument('--export',type=Path)
    args=parser.parse_args()
    cfg=TrainConfig(**json.loads(args.config.read_text())) if args.config else TrainConfig()
    if args.prepare:prepare_split(cfg)
    elif args.test:print(test_checkpoint(args.test))
    elif args.export:print(export_checkpoint(args.export))
    else:train(cfg)
