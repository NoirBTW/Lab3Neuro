from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'experiments/notebooks/lab3.ipynb'
notebook=nbformat.read(path,as_version=4)
def save(**kwargs):nbformat.write(notebook,path)
client=NotebookClient(notebook,timeout=3600,kernel_name='neuro-lab',resources={'metadata':{'path':str(ROOT)}},on_cell_executed=save)
try:client.execute()
finally:nbformat.write(notebook,path)
print('Notebook completed',flush=True)
