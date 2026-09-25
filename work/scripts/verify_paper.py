"""Check required paper content and render desktop/mobile locally."""
from pathlib import Path
import functools
import hashlib
import http.server
import json
import os
import re
import threading
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'work/paper'
CACHE=ROOT/'work/outputs/capstone_cache'
CACHE.mkdir(parents=True,exist_ok=True)
raw=(PAPER/'index.html').read_text()
soup=BeautifulSoup(raw,'html.parser')
required=['abstract','problem','data','method','results','limitations','recommendations','reproducibility','credit']
assert [s['id'] for s in soup.select('article > section')]==required
assert len(soup.select('#recommendations .recommendation'))==10
assert soup.select_one('#credit a')['href']=='https://flyrank.ai'
abstract=soup.select_one('#abstract p').get_text()
sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])',abstract)
assert len(sentences)==5,(len(sentences),sentences)
meta=json.loads((PAPER/'publication.json').read_text())
assert meta['metrics_sha256']==hashlib.sha256((ROOT/'work/outputs/capstone_metrics.json').read_bytes()).hexdigest()
assert 'hf_' not in raw and not re.search(r'(ghp_|github_pat_)[A-Za-z0-9_]+',raw)
for node,attr in [('img','src'),('link','href')]:
 for element in soup.select(node):
  value=element.get(attr,'')
  if value and not value.startswith(('http','#')):assert (PAPER/value).is_file(),value
for link in soup.select('a[href^="#"]'):
 target=link['href'][1:]
 if target:assert soup.find(id=target),target
print('Content checks passed: nine sections, five-sentence abstract, ten recommendations, linked data credit, local assets and current receipts.',flush=True)
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(ROOT/'work/outputs/publish_auth/browsers'))
from playwright.sync_api import sync_playwright
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
handler=functools.partial(Quiet,directory=str(PAPER))
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
url=f'http://127.0.0.1:{server.server_port}/'
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 for name,width,height in [('desktop',1440,1000),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  response=page.goto(url,wait_until='networkidle')
  assert response.status==200
  assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),name+' overflow'
  assert page.locator('article > section').count()==9
  assert page.locator('img').evaluate_all('(imgs)=>imgs.every(i=>i.complete && i.naturalWidth>0)')
  assert not errors,errors
  page.screenshot(path=str(CACHE/f'paper-{name}.png'),full_page=False)
  page.locator('#results').scroll_into_view_if_needed()
  page.screenshot(path=str(CACHE/f'paper-{name}-results.png'),full_page=False)
  print(f'{name}: rendering, images, navigation targets and no-overflow checks passed.',flush=True)
  page.close()
 browser.close()
server.shutdown()
