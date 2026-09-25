"""Execute and save capstone deliverables with fail-fast behavior."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'work/outputs/capstone_cache/jupyter'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'work/outputs/capstone_cache/ipython'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/outputs/capstone_cache/matplotlib'))
for name in ['w05_model','w06_validation_audit','w07_action_playbook','capstone']:
    p=ROOT/'work/notebooks'/f'{name}.ipynb'
    n=nbformat.read(p,as_version=4)
    NotebookClient(n,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(p.parent)}}).execute()
    for c in n.cells:
        if c.cell_type=='code':
            assert c.execution_count is not None
            assert not any(o.output_type=='error' for o in c.outputs)
    nbformat.validate(n)
    nbformat.write(n,p)
    print('Executed and saved:',name,flush=True)
