"""Publish the committed paper, verify it, then commit its real URL.

Uses an existing GitHub CLI login for the repository owner, GH_TOKEN/GITHUB_TOKEN,
or the ignored local publishing token. Never prints credentials or changes the
user's active account. Requires a clean committed research tree before publishing.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
OWNER='tanzimul3islam'
REPOSITORY=OWNER+'/flyrank-ml-internship-starter'
AUTH=ROOT/'work/outputs/publish_auth'
DEFAULT_URL='https://tanzimul3islam.github.io/flyrank-ml-internship-starter/'


def read_token():
    token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if not token:
        try:
            result=subprocess.run(['gh','auth','token','--hostname','github.com','--user',OWNER],
                                  capture_output=True,text=True,check=True)
            token=result.stdout.strip()
        except (FileNotFoundError,subprocess.CalledProcessError):pass
    if not token and (AUTH/'token').is_file():token=(AUTH/'token').read_text().strip()
    if not token:raise SystemExit('No GitHub publishing login. Authenticate the repository-owner account and rerun.')
    return token


def api(token,path,method='GET',body=None,missing_ok=False):
    req=Request('https://api.github.com'+path,method=method,
                data=json.dumps(body).encode() if body is not None else None,
                headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                         'Content-Type':'application/json','X-GitHub-Api-Version':'2026-03-10'})
    try:
        with urlopen(req,timeout=40) as r:
            raw=r.read()
            return json.loads(raw) if raw else {}
    except HTTPError as e:
        if missing_ok and e.code==404:return None
        raise SystemExit(f'GitHub API {method} {path} returned HTTP {e.code}; check account permissions or workflow status. No credential was printed.') from None


def git(args,token=None):
    env=os.environ.copy()
    prefix=[]
    if token:
        AUTH.mkdir(parents=True,exist_ok=True)
        helper=AUTH/'git_askpass.py'
        helper.write_text('#!'+sys.executable+'\nimport os,sys\nprint("x-access-token" if "Username" in sys.argv[1] else os.environ["FLYRANK_PUBLISH_TOKEN"])\n')
        helper.chmod(0o700)
        env.update(GIT_ASKPASS=str(helper),GIT_TERMINAL_PROMPT='0',FLYRANK_PUBLISH_TOKEN=token)
        prefix=['-c','credential.helper=']
    result=subprocess.run(['git',*prefix,*args],cwd=ROOT,env=env,capture_output=True,text=True)
    if result.returncode:
        message=(result.stderr or result.stdout).strip()
        if token:message=message.replace(token,'[redacted]')
        raise SystemExit('Git command failed: '+message)
    return result.stdout.strip()


def verify_live(url,expected):
    try:
        with urlopen(Request(url+'publication.json',headers={'Cache-Control':'no-cache'}),timeout=20) as r:
            live=json.load(r)
        if live!=expected:return False
        with urlopen(url,timeout=20) as r:page=r.read().decode()
        return expected['title'] in page and 'https://flyrank.ai' in page
    except (HTTPError,URLError,TimeoutError,ValueError):return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-access',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    expected=json.loads((ROOT/'work/paper/publication.json').read_text())
    token=read_token()
    repo=api(token,'/repos/'+REPOSITORY)
    if not repo.get('permissions',{}).get('push'):
        identity=api(token,'/user').get('login','unknown')
        raise SystemExit(f'Account {identity} cannot push to {REPOSITORY}. Use the repository-owner login; no remote change made.')
    print('Repository write access verified.',flush=True)
    if args.check_access:return
    settings=api(token,'/repos/'+REPOSITORY+'/pages',missing_ok=True)
    if not args.verify_only:
        changes=git(['status','--porcelain'])
        if changes:raise SystemExit('Commit the reviewed research artifacts before publishing. Uncommitted files remain.')
        if settings is None:
            settings=api(token,'/repos/'+REPOSITORY+'/pages',method='POST',body={'build_type':'workflow'})
        elif settings.get('build_type')!='workflow':
            api(token,'/repos/'+REPOSITORY+'/pages',method='PUT',body={'build_type':'workflow'})
        git(['push','origin','main'],token)
        # Push normally starts the workflow; dispatch also supports a no-op/retry push.
        for _ in range(6):
            workflow=api(token,'/repos/'+REPOSITORY+'/actions/workflows/publish-paper.yml',missing_ok=True)
            if workflow:break
            time.sleep(5)
        if workflow:
            api(token,'/repos/'+REPOSITORY+'/actions/workflows/publish-paper.yml/dispatches',method='POST',body={'ref':'main'})
        print('Paper pushed; waiting for the matching Pages publication.',flush=True)
    settings=api(token,'/repos/'+REPOSITORY+'/pages',missing_ok=True) or {}
    url=(settings.get('html_url') or DEFAULT_URL).rstrip('/')+'/'
    if not url.startswith('https://'):raise SystemExit('Pages has not provided an HTTPS URL yet.')
    for attempt in range(40):
        if verify_live(url,expected):break
        if args.verify_only:raise SystemExit('The matching paper is not live yet; no submission URL written.')
        if attempt%3==0:print('Awaiting matching live publication...',flush=True)
        time.sleep(15)
    else:raise SystemExit('Deployment has not completed. Check GitHub Actions and rerun --verify-only. No unverified URL recorded.')
    # Only now is it truthful to record a deployed paper URL.
    (ROOT/'submission/paper_url.txt').write_text(url+'\n')
    p=ROOT/'work/notebooks/capstone.ipynb'
    n=json.loads(p.read_text())
    for cell in n['cells']:
        if cell['cell_type']=='markdown':
            source=''.join(cell['source'])
            source=source.replace('- [ ] Public deployment verified and its exact URL written',
                                  '- [x] Public deployment verified and its exact URL written')
            cell['source']=source.splitlines(keepends=True)
    p.write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n')
    git(['add','submission/paper_url.txt','work/notebooks/capstone.ipynb'])
    staged=git(['diff','--cached','--name-only'])
    if staged:
        git(['commit','-m','Record verified live research paper URL'])
        git(['push','origin','main'],token)
    print('Verified live paper: '+url,flush=True)
    print('Mandatory submission/paper_url.txt committed and pushed.',flush=True)


if __name__=='__main__':main()
