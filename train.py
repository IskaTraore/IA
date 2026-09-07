"""Entraînement YOLOv8n sur le dataset FieldPlant (CPU, reprise automatique).

Usage :
    python train.py            # entraîne (ou reprend) jusqu'à epochs
    python train.py --epochs N # surcharge le nombre d'époques

Le modèle reprend automatiquement depuis runs/detect/train*/weights/last.pt
si un entraînement précédent a été interrompu.
"""
import argparse
import glob
import os

# Le project doit être un chemin ABSOLU : sinon Ultralytics le résout relativement à
# son runs_dir global (settings.json) et crée runs/detect/runs/detect/... imbriqué.
RUNS_DIR = os.path.abspath("runs/detect")

from ultralytics import YOLO

# --- Configuration ---
DATA_YAML = os.path.abspath("datasets/fieldplant/data.yaml")
MODEL = "yolov8n.pt"       # nano : adapté CPU ; essayer yolov8s.pt pour comparer
IMGSZ = 640                # taille d'entrée
BATCH = 16
EPOCHS = 60                # early stopping à patience=12
PATIENCE = 12
SEED = 42
DEVICE = "cpu"             # pas de GPU détecté sur cette machine
WORKERS = 2

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--smoke", action="store_true",
                    help="mini-dataset de 40 images, 1 époque : valide le pipeline")
args = parser.parse_args()

# --- Smoke test : mini dataset de 40 images ---
if args.smoke:
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="fieldplant_smoke_")
    for split in ("train", "valid"):
        src_imgs = f"datasets/fieldplant/{split}/images"
        dst = os.path.join(tmp, split)
        os.makedirs(os.path.join(dst, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
        for f in sorted(os.listdir(src_imgs))[:20]:
            stem, ext = os.path.splitext(f)
            shutil.copy(os.path.join(src_imgs, f), os.path.join(dst, "images", f))
            shutil.copy(os.path.join(src_imgs.replace("images", "labels"), stem + ".txt"),
                        os.path.join(dst, "labels", stem + ".txt"))
    import yaml
    full = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))
    smoke_yaml = {
        "path": os.path.abspath(tmp),
        "train": "train/images",
        "val": "valid/images",
        "nc": full["nc"],
        "names": full["names"],
    }
    smoke_path = os.path.join(tmp, "data.yaml")
    yaml.dump(smoke_yaml, open(smoke_path, "w", encoding="utf-8"))
    print(f"Smoke test sur : {smoke_path}")
    model = YOLO(MODEL)
    model.train(
        data=smoke_path, epochs=1, imgsz=IMGSZ, batch=4, seed=SEED,
        device=DEVICE, workers=0, project=RUNS_DIR, name="smoke",
        plots=False, verbose=True,
    )
    shutil.rmtree(tmp)
    print("Smoke test réussi.")
    raise SystemExit(0)

# --- Reprise si un entraînement existe déjà ---
existing = sorted(glob.glob(os.path.join(RUNS_DIR, "train*", "weights", "last.pt")))
resume = existing[-1] if existing else None

if resume and os.path.exists(resume):
    print(f"Reprise de l'entraînement depuis : {resume}")
    model = YOLO(resume)
    model.train(resume=True, epochs=args.epochs)
else:
    print(f"Nouvel entraînement : {MODEL} sur {DATA_YAML}")
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        device=DEVICE,
        workers=WORKERS,
        project=RUNS_DIR,
        name="train",
        exist_ok=False,       # créer train2, train3... si reprise multiple
        plots=True,           # génère confusion matrix, PR curves...
    )

print("Entraînement terminé.")