"""Decisions made by visual inspection of all candidates before any training."""
import csv,json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
REJECT={
 '129745856':'insect too small in a wide colony view','129746000':'insect too small in a wide colony view',
 '53889934':'illustration, not a photograph','25842165':'isolated extreme head detail',
 '64927950':'caterpillar','118622130':'caterpillar','64927537':'caterpillar',
 '63168428':'insect cut off by image boundary','67729199':'dark tiny silhouette',
 '15268058':'empty exuvia','7352494':'emergence stage, not a fully developed adult',
 '113141457':'insect too small in wide pond view','113141464':'insect too small in wide pond view',
 '113113952':'insect too small in wide pond view','151166706':'isolated extreme head detail',
 '188138560':'ambiguous scene with bee and ladybird','117787013':'larva',
 '87953841':'rendered illustration','27752320':'insect too small'}
def curate():
    dest=ROOT/'experiments/data';source=dest/'sources.csv'
    candidates=dest/'candidates.csv'
    rows=list(csv.DictReader((candidates if candidates.exists() else source).open(encoding='utf-8')))
    if not (dest/'candidates.csv').exists():shutil.copyfile(source,dest/'candidates.csv')
    accepted=[];rejected=[]
    for r in rows:
        key=Path(r['path']).stem
        if key in REJECT:rejected.append(dict(r,reason=REJECT[key]))
        else:accepted.append(r)
    keep={r['path'] for r in accepted}
    # Move excluded and stale downloaded candidates into ignored local QA storage.
    for p in (dest/'raw').glob('*/*'):
        if p.relative_to(ROOT).as_posix() not in keep:
            target=ROOT/'report_work/rejected'/p.parent.name/p.name
            assert p.resolve().is_relative_to(ROOT.resolve()) and target.resolve().is_relative_to(ROOT.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.move(str(p),str(target))
    with source.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(accepted)
    with (dest/'rejected.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rejected[0]));w.writeheader();w.writerows(rejected)
    print(Counter(r['label'] for r in accepted))
    hashes=[];near=[]
    for r in accepted:
        im=np.asarray(Image.open(ROOT/r['path']).convert('L').resize((9,8)))
        h=(im[:,1:]>im[:,:-1]).reshape(-1)
        for prev,ph in hashes:
            distance=int(np.count_nonzero(h!=ph))
            if distance<=5:near.append(dict(a=prev['path'],b=r['path'],hamming=distance,same_author=prev['author']==r['author']))
        hashes.append((r,h))
    (dest/'duplicate_review.json').write_text(json.dumps(near,indent=2))
    print('Near duplicates',near)
    fig,axs=plt.subplots(3,3,figsize=(9,8))
    for i,c in enumerate(sorted({r['label'] for r in accepted})):
        for ax,r in zip(axs[i],[r for r in accepted if r['label']==c][:3]):
            ax.imshow(Image.open(ROOT/r['path']));ax.set_title(c+' / '+Path(r['path']).stem);ax.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'experiments/reports/figures/examples.png',dpi=140);plt.close(fig)
if __name__=='__main__':curate()
