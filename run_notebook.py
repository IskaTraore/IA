"""Exécute notebook_finetuning.ipynb via nbclient (mode execute).

- Chaque cellule idempotente : téléchargement/split/entraînement sautés si déjà faits.
- L'entraînement reprend depuis runs/detect/<exp>/weights/last.pt (resume).
- Timeout par cellule : 480 s — au dépassement, la progression est sauvegardée
  et le script s'arrête proprement ; il suffit de relancer.
"""
import asyncio
import os
import sys

# Nécessaire sous Windows : ZMQ + Proactor loop ne cohabitent pas bien
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Console en UTF-8 (évite les erreurs d'encodage cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# S'assurer que le module partagé fieldplant/ est importable (CWD = racine projet)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError

os.environ.setdefault("RUN_MODE", "cpu_subset")
os.environ.setdefault("EXP_NAME", "yolo_cpu")
os.environ.setdefault("IMGSZ", "320")
os.environ.setdefault("EPOCHS", "20")
os.environ.setdefault("FRACTION", "0.12")

NOTEBOOK = "notebook_finetuning.ipynb"
KERNEL = "fieldplant"
CELL_TIMEOUT = 480

nb = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(nb, timeout=CELL_TIMEOUT, kernel_name=KERNEL)

print("Kernel :", KERNEL, "| mode :", os.environ["RUN_MODE"], "| timeout cellule :", CELL_TIMEOUT)

try:
    client.execute()
    print("Notebook entierement execute.")
    nbformat.write(nb, NOTEBOOK)
    sys.exit(0)
except CellTimeoutError:
    nbformat.write(nb, NOTEBOOK)
    print(f"Timeout cellule (> {CELL_TIMEOUT}s) - progression sauvegardee. Relancer pour continuer.")
    sys.exit(0)
except Exception as e:
    nbformat.write(nb, NOTEBOOK)
    print(f"Erreur : {type(e).__name__}: {e}")
    sys.exit(1)