"""Collect individual openly licensed photographs, never a packaged dataset."""
from pathlib import Path
import urllib.request, urllib.parse
import json, csv, html, re, time, hashlib
from PIL import Image
from io import BytesIO

ROOT=Path(__file__).resolve().parents[1]
QUERIES={'butterfly':'intitle:"Danaus plexippus"',
         'dragonfly':'intitle:"Anax imperator"',
         'ladybird':'intitle:"Coccinella septempunctata"'}
HEADERS={'User-Agent':'Lab3Neuro/1.0 (educational image collection; github.com/NoirBTW/Lab3Neuro)'}
def fetch(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=40).read()
        except Exception:
            if attempt==3:raise
            time.sleep(2+attempt*2)
def plain(value):return html.unescape(re.sub('<[^>]+>','',value)).strip()

def collect():
    dest=ROOT/'experiments/data'; rows=[]
    for label,query in QUERIES.items():
        folder=dest/'raw'/label;folder.mkdir(parents=True,exist_ok=True)
        params=dict(action='query',format='json',generator='search',gsrsearch=query+' filetype:bitmap',
                    gsrnamespace=6,gsrlimit=50,prop='imageinfo',iiprop='url|extmetadata|size',iiurlwidth=512)
        data=json.loads(fetch('https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)))
        params['gsroffset']=50
        extra=json.loads(fetch('https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)))
        data['query']['pages'].update(extra.get('query',{}).get('pages',{}))
        (dest/(label+'_search.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        count=0
        for page in sorted(data['query']['pages'].values(),key=lambda p:p['index']):
            if 'imageinfo' not in page:continue
            info=page['imageinfo'][0];meta=info.get('extmetadata',{})
            get=lambda k:plain(meta.get(k,{}).get('value',''))
            license=get('LicenseShortName');title=page['title']
            if not ('CC BY' in license.upper() or license in ['CC0','Public domain']):continue
            if min(info['width'],info['height'])<300:continue
            if any(w in title.lower() for w in ['larva','pupa','egg','caterpillar','drawing','illustration','map','distribution','museum','specimen','dead','mhnt','exuvi','rupsen','markasy','pheasant']):continue
            name=f'{page["pageid"]}.jpg';target=folder/name
            url=info.get('thumburl',info['url'])
            try:
                if not target.exists():
                    content=fetch(url)
                    with Image.open(BytesIO(content)) as im:im.verify()
                    target.write_bytes(content)
                with Image.open(target) as im: w,h=im.size
            except Exception as e:
                print('Skip',title,type(e).__name__,flush=True);continue
            rows.append(dict(path=target.relative_to(ROOT).as_posix(),label=label,title=title,
                author=get('Artist'),license=license,license_url=get('LicenseUrl'),
                source_page=info['descriptionurl'],original_url=info['url'],download_url=url,
                collected_at='2026-09-07',width=w,height=h,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                modification='Wikimedia server thumbnail; no local pixel editing'))
            count+=1;print(label,count,title,flush=True)
            if count>=45:break
        assert count>=30,(label,count)
    with (dest/'sources.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Collected',len(rows),flush=True)
if __name__=='__main__':collect()
