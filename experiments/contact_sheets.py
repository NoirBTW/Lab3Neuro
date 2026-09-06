from pathlib import Path
import csv
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'experiments/data/sources.csv').open(encoding='utf-8')))
out=ROOT/'report_work/contacts';out.mkdir(parents=True,exist_ok=True)
for label in sorted({r['label'] for r in rows}):
    subset=[r for r in rows if r['label']==label]
    for page in range((len(subset)+19)//20):
        canvas=Image.new('RGB',(1100,1000),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(subset[page*20:page*20+20]):
            im=Image.open(ROOT/r['path']).convert('RGB');im.thumbnail((210,210))
            x=i%5*220;y=i//5*250
            canvas.paste(im,(x+(210-im.width)//2,y));draw.text((x+5,y+215),Path(r['path']).stem,fill='black')
        canvas.save(out/f'{label}_{page+1}.jpg')
