"""Génère le notebook de finetuning (livrable du projet) via nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells = []

# ---------------------------------------------------------------- Markdown
cells.append(nbf.v4.new_markdown_cell("""# Détection de maladies foliaires par YOLO — Notebook de finetuning

**Objectif :** concevoir, entraîner et évaluer un détecteur d'objets YOLO capable de localiser
(bounding box) et de classer des maladies foliaires sur des photos de plantes prises en
conditions de terrain (champ, serre, jardin).

**Dataset :** [FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant)
— 5 156 images annotées (bounding boxes) par des pathologistes végétaux, 27 classes
(manioc, maïs, tomate), licence CC BY 4.0.

> Citation : Moupojou, E., et al. *FieldPlant: A Dataset of Field Plant Images for Plant
> Disease Detection and Classification With Deep Learning.* IEEE Access, vol. 11, 2023.

**Déroulé du notebook :**
1. Installation et configuration (GPU Colab auto-détecté, repli CPU)
2. Téléchargement du dataset (Roboflow, clé API gratuite)
3. Découpage train / valid / test
4. Exploration du dataset
5. Entraînement YOLOv8n (fine-tuning) — *reprise automatique si interrompu*
6. Évaluation sur le split test (mAP, précision, rappel, F1, matrices, courbes PR)
7. Exemples qualitatifs (bonnes détections, faux positifs, faux négatifs)
8. Temps d'inférence et taille du modèle"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Installation et configuration

Deux modes d'exécution :
- **`colab_gpu`** (défaut) : entraînement complet sur GPU Google Colab (recommandé, ~1 min/époque).
- **`cpu_subset`** : entraînement réduit sur CPU local (sous-échantillonnage + imgsz réduit)
  pour valider le pipeline sur une machine sans GPU.

Le mode se règle via la variable d'environnement `RUN_MODE` ou directement ci-dessous."""))

cells.append(nbf.v4.new_code_cell("""import os
import sys
import glob
import shutil
import time
from collections import defaultdict

# --- Installation automatique (Colab) ---
try:
    import ultralytics
except ImportError:
    get_ipython().system("pip install -q ultralytics")

# Module partagé fieldplant (téléchargement + split) : présent en local,
# téléchargé sur Colab depuis le dépôt du projet.
try:
    import fieldplant
    import fieldplant.data
except ImportError:
    get_ipython().system("pip install -q roboflow pyyaml")
    # NB : adapter l'URL au dépôt contenant ce projet
    get_ipython().system(
        "wget -q https://github.com/IskaTraore/IA/archive/refs/heads/main.zip -O repo.zip"
        " && unzip -o -q repo.zip '*/fieldplant/*' -d /tmp/fp"
        " && cp -r /tmp/fp/*/fieldplant . && rm -rf repo.zip /tmp/fp")
    try:
        import fieldplant
        import fieldplant.data
    except ImportError as e:
        raise ImportError(
            "Module fieldplant introuvable : copie le dossier fieldplant/ du projet "
            "dans le répertoire courant du notebook (cf. README).") from e

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
import torch
import ultralytics

print("Python  :", sys.version.split()[0])
print("torch   :", torch.__version__, "| ultralytics:", ultralytics.__version__)
print("GPU CUDA:", torch.cuda.is_available(), "| MPS:", torch.backends.mps.is_available())

# --- Mode d'exécution ---
RUN_MODE = os.environ.get("RUN_MODE", "colab_gpu")   # "colab_gpu" ou "cpu_subset"
print("Mode d'exécution :", RUN_MODE)"""))

cells.append(nbf.v4.new_code_cell("""# --- Paramètres (surchargables par variables d'environnement) ---
DATA_YAML = "datasets/fieldplant/data.yaml"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Chemin ABSOLU : sinon Ultralytics résout le project relativement à son runs_dir
# global (settings.json) et crée runs/detect/runs/detect/... imbriqué.
RUNS_DIR = os.path.abspath("runs/detect")

if RUN_MODE == "cpu_subset":
    # Config allégée pour CPU : sous-échantillonnage + petite résolution
    MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")
    IMGSZ = int(os.environ.get("IMGSZ", "320"))
    EPOCHS = int(os.environ.get("EPOCHS", "20"))
    BATCH = int(os.environ.get("BATCH", "16"))
    FRACTION = float(os.environ.get("FRACTION", "0.12"))   # ~500 images train
    EXP_NAME = os.environ.get("EXP_NAME", "yolo_cpu")
    WORKERS = 2
else:
    # Config complète pour GPU (Colab)
    MODEL_NAME = os.environ.get("MODEL_NAME", "yolov8n.pt")
    IMGSZ = int(os.environ.get("IMGSZ", "640"))
    EPOCHS = int(os.environ.get("EPOCHS", "60"))
    BATCH = int(os.environ.get("BATCH", "16"))
    FRACTION = float(os.environ.get("FRACTION", "1.0"))    # dataset complet
    EXP_NAME = os.environ.get("EXP_NAME", "yolo_gpu")
    WORKERS = 8

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Modèle    : {MODEL_NAME}")
print(f"imgsz     : {IMGSZ}  |  époques : {EPOCHS}  |  batch : {BATCH}")
print(f"fraction  : {FRACTION}  |  device : {DEVICE}  |  exp : {EXP_NAME}")"""))

# ---------------------------------------------------------------- Dataset
cells.append(nbf.v4.new_markdown_cell("""## 2. Téléchargement du dataset (Roboflow)

Le dataset **FieldPlant** est téléchargé au format YOLOv8 depuis Roboflow Universe
via le module partagé `fieldplant.data` (idempotent).
Une clé API Roboflow gratuite est nécessaire : `app.roboflow.com/settings/api`.
La clé est lue depuis la variable d'environnement `ROBOFLOW_API_KEY` ou `.env`.

> ℹ️ Sur Colab : colle ta clé quand le champ s'affiche (getpass), ou définis
> `ROBOFLOW_API_KEY` dans les secrets du notebook."""))

cells.append(nbf.v4.new_code_cell("""DATA_DIR = fieldplant.data.DATA_DIR  # "datasets/fieldplant"

# Clé API : variable d'environnement -> .env -> saisie interactive (Colab)
if not os.environ.get("ROBOFLOW_API_KEY"):
    try:
        fieldplant.data.load_roboflow_api_key()
    except SystemExit:
        import getpass
        os.environ["ROBOFLOW_API_KEY"] = getpass.getpass("Clé API Roboflow : ")

fieldplant.data.download_dataset()
n = len(os.listdir(os.path.join(DATA_DIR, "train", "images")))
print(f"Dataset prêt : {DATA_DIR} ({n} images train).")"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Découpage train / valid / test

La version Roboflow du dataset ne fournit qu'un split `train` (5 156 images).
On crée donc des splits **stratifiés 80/10/10** (seed fixe 42) en s'assurant que
chaque classe est représentée dans chaque split — via `fieldplant.data.stratified_split`."""))

cells.append(nbf.v4.new_code_cell("""# Split stratifié 80/10/10 (seed 42) — logique partagée dans fieldplant.data
fieldplant.data.stratified_split(DATA_DIR)

# Récapitulatif
for split in ("train", "valid", "test"):
    n_img = len(os.listdir(os.path.join(DATA_DIR, split, "images")))
    n_lbl = len(os.listdir(os.path.join(DATA_DIR, split, "labels")))
    print(f"{split:6s} : {n_img} images / {n_lbl} labels")"""))

# ---------------------------------------------------------------- Exploration
cells.append(nbf.v4.new_markdown_cell("""## 4. Exploration du dataset

- Distribution des 27 classes (nombre de boîtes par classe)
- Exemples d'images avec leurs annotations"""))

cells.append(nbf.v4.new_code_cell("""with open(DATA_YAML, encoding="utf-8") as f:
    data = yaml.safe_load(f)
names = data["names"]
print(f"{len(names)} classes :")
print(" - " + "\\n - ".join(names))

# Distribution des instances par classe
counts = defaultdict(int)
for split in ("train", "valid", "test"):
    lbl_dir = os.path.join(DATA_DIR, split, "labels")
    for lf in os.listdir(lbl_dir):
        with open(os.path.join(lbl_dir, lf)) as f:
            for line in f:
                line = line.strip()
                if line:
                    counts[int(line.split()[0])] += 1

df_counts = pd.DataFrame({"classe": [names[c] for c in range(len(names))],
                          "instances": [counts.get(c, 0) for c in range(len(names))]})
df_counts = df_counts.sort_values("instances", ascending=False).reset_index(drop=True)
print(df_counts.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(df_counts["classe"], df_counts["instances"], color="#4C72B0")
ax.invert_yaxis()
ax.set_xlabel("Nombre d'instances annotées")
ax.set_title("Distribution des classes — FieldPlant (train+valid+test)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "distribution_classes.png"), dpi=150)
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Exemples d'images annotées (grille 3x2)
from PIL import Image

def draw_boxes(img_path, label_path, ax):
    img = Image.open(img_path)
    ax.imshow(img)
    w, h = img.size
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c, cx, cy, bw, bh = map(float, line.split())
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                rect = mpatches.Rectangle((x, y), bw * w, bh * h,
                                          linewidth=1.5, edgecolor="#E15759",
                                          facecolor="none")
                ax.add_patch(rect)
                ax.text(x, max(0, y - 3), names[int(c)], fontsize=7, color="#E15759")
    ax.axis("off")

sample_imgs = sorted(os.listdir(os.path.join(DATA_DIR, "train", "images")))[:6]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, fname in zip(axes.ravel(), sample_imgs):
    stem, _ = os.path.splitext(fname)
    draw_boxes(os.path.join(DATA_DIR, "train", "images", fname),
               os.path.join(DATA_DIR, "train", "labels", stem + ".txt"), ax)
plt.suptitle("Exemples d'images annotées (train)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "exemples_annotes.png"), dpi=150)
plt.show()"""))

# ---------------------------------------------------------------- Training
cells.append(nbf.v4.new_markdown_cell("""## 5. Entraînement YOLOv8n (fine-tuning)

On part des poids pré-entraînés COCO (`yolov8n.pt`) et on fine-tune sur FieldPlant :
- **imgsz** : résolution d'entrée (640 GPU / 320 CPU)
- **epochs** : 60 GPU / 20 CPU, avec **early stopping** (patience 15)
- **fraction** : 100 % du dataset (GPU) ou ~12 % (~500 images, CPU)
- **cache** : images en RAM pour accélérer (ok avec 16 Go)
- **plots** : courbes d'apprentissage, matrice de confusion, courbes PR générées
  automatiquement dans `runs/detect/<exp>/`

> 🔄 **Reprise automatique** : si l'entraînement est interrompu (timeout, coupure),
> la cellule reprend depuis `last.pt` au prochain passage. Si `best.pt` existe,
> l'entraînement est considéré terminé."""))

cells.append(nbf.v4.new_code_cell("""from ultralytics import YOLO

exp_dir = os.path.join(RUNS_DIR, EXP_NAME)
best_paths = glob.glob(os.path.join(exp_dir, "weights", "best.pt"))
last_paths = glob.glob(os.path.join(exp_dir, "weights", "last.pt"))

if best_paths:
    model = YOLO(best_paths[-1])
    print(f"Entraînement déjà terminé — chargement de {best_paths[-1]}")
elif last_paths:
    model = YOLO(last_paths[-1])
    print(f"Reprise de l'entraînement depuis {last_paths[-1]} (époques totales : {EPOCHS})")
    model.train(resume=True, epochs=EPOCHS)
else:
    model = YOLO(MODEL_NAME)
    print(f"Nouvel entraînement : {MODEL_NAME} sur {DATA_YAML} "
          f"({EPOCHS} époques, imgsz={IMGSZ}, fraction={FRACTION})")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=15,
        seed=42,
        cache=True,
        fraction=FRACTION,
        device=DEVICE,
        workers=WORKERS,
        project=RUNS_DIR,
        name=EXP_NAME,
        plots=True,
    )
print("Modèle prêt :", best_paths[-1] if best_paths else "voir runs/detect/" + EXP_NAME)"""))

# ---------------------------------------------------------------- Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 6. Évaluation sur le split test

Métriques standard de détection (calculées par Ultralytics) :
- **mAP@0.5** et **mAP@0.5:0.95**
- **Précision / Rappel / F1** par classe
- **Matrice de confusion** et **courbes Précision-Rappel** par classe"""))

cells.append(nbf.v4.new_code_cell("""# Évaluation sur le split test
results = model.val(data=DATA_YAML, split="test", batch=BATCH, plots=True, device=DEVICE)

m = results.box
print("\\n=== Métriques globales (test) ===")
print(f"mAP@0.5      : {m.map50:.4f}")
print(f"mAP@0.5:0.95 : {m.map:.4f}")
print(f"Précision    : {m.mp:.4f}")
print(f"Rappel       : {m.mr:.4f}")

# Métriques par classe
per_class = pd.DataFrame({
    "classe": [names[int(i)] for i in m.ap_class_index],
    "précision": m.p,
    "rappel": m.r,
    "mAP@0.5": m.ap50,
    "mAP@0.5:0.95": m.ap,
})
per_class["F1"] = 2 * per_class["précision"] * per_class["rappel"] / (
    per_class["précision"] + per_class["rappel"] + 1e-9)
per_class = per_class.sort_values("mAP@0.5", ascending=False).reset_index(drop=True)
print("\\n=== Métriques par classe ===")
print(per_class.to_string(index=False))
per_class.to_csv(os.path.join(FIG_DIR, "metriques_par_classe.csv"), index=False)"""))

cells.append(nbf.v4.new_code_cell("""# Affichage des courbes générées par Ultralytics (val)
val_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, EXP_NAME, "*.png")))
val_dirs += sorted(glob.glob(os.path.join(RUNS_DIR, EXP_NAME, "val*", "*.png")))

def show_img(path, title, ax):
    ax.imshow(plt.imread(path))
    ax.set_title(title)
    ax.axis("off")

targets = ["PR_curve", "confusion_matrix", "F1_curve", "results"]
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
plotted = []
for ax, key in zip(axes.ravel(), targets):
    found = [p for p in val_dirs if key in os.path.basename(p)]
    if found:
        show_img(found[0], key, ax)
        plotted.append(found[0])
plt.suptitle(f"Courbes d'évaluation — {EXP_NAME}", fontsize=15)
plt.tight_layout()
plt.show()
print("Fichiers :", plotted)

# Copie des figures clés dans figures/
for key in ("PR_curve", "confusion_matrix"):
    for p in val_dirs:
        if key in os.path.basename(p):
            shutil.copy(p, os.path.join(FIG_DIR, os.path.basename(p)))
            break"""))

# ---------------------------------------------------------------- Inference demo
cells.append(nbf.v4.new_markdown_cell("""## 7. Exemples qualitatifs

Prédictions sur des images du split test : bonnes détections, faux positifs,
faux négatifs — à commenter dans le rapport."""))

cells.append(nbf.v4.new_code_cell("""# Inférence sur 6 images test
test_imgs = sorted(os.listdir(os.path.join(DATA_DIR, "test", "images")))[:6]
paths = [os.path.join(DATA_DIR, "test", "images", f) for f in test_imgs]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, p in zip(axes.ravel(), paths):
    res = model.predict(p, conf=0.25, device=DEVICE, verbose=False)[0]
    img = plt.imread(p)
    ax.imshow(img)
    h, w = img.shape[:2]
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor="#4C72B0", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 3), f"{names[cls]} {conf:.2f}", fontsize=7, color="#4C72B0")
    ax.axis("off")
    ax.set_title(os.path.basename(p)[:28], fontsize=8)
plt.suptitle("Prédictions sur le split test (conf >= 0.25)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "predictions_test.png"), dpi=150)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## 8. Temps d'inférence et taille du modèle

Compromis précision / vitesse, utile pour discuter nano vs small."""))

cells.append(nbf.v4.new_code_cell("""# Benchmark inférence sur 20 images test
bench_imgs = sorted(os.listdir(os.path.join(DATA_DIR, "test", "images")))[:20]
bench_paths = [os.path.join(DATA_DIR, "test", "images", f) for f in bench_imgs]

# Warm-up
_ = model.predict(bench_paths[0], conf=0.25, device=DEVICE, verbose=False)

times = []
for p in bench_paths:
    t0 = time.perf_counter()
    _ = model.predict(p, conf=0.25, device=DEVICE, verbose=False)
    times.append((time.perf_counter() - t0) * 1000)

mean_ms = float(np.mean(times))
n_params = sum(p.numel() for p in model.model.parameters())
size_mb = os.path.getsize(best_paths[-1] if best_paths
                          else glob.glob(os.path.join(exp_dir, "weights", "*.pt"))[-1]) / 1e6

print(f"Temps d'inférence moyen : {mean_ms:.1f} ms/image  ({1000 / mean_ms:.1f} FPS)")
print(f"Paramètres              : {n_params / 1e6:.2f} M")
print(f"Taille du modèle (.pt)  : {size_mb:.1f} Mo")
print(f"Device                  : {DEVICE}")"""))

cells.append(nbf.v4.new_markdown_cell("""## 9. Conclusion et pistes

**Résumé attendu (à compléter dans le rapport) :**
- mAP@0.5 et mAP@0.5:0.95 sur le split test
- Classes les mieux / moins bien détectées (voir tableau par classe)
- Erreurs typiques : confusion entre maladies visuellement proches, symptômes minuscules…

**Pistes d'amélioration :**
- Entraînement complet sur GPU (Colab, ~60 époques, imgsz 640)
- Comparaison YOLOv8n vs YOLOv8s (précision vs vitesse)
- Augmentation ciblée pour les classes sous-représentées
- Test en conditions réelles (photos smartphone, autres cultures)"""))

nb.cells = cells
nbf.write(nb, "notebook_finetuning.ipynb")
print("Notebook généré : notebook_finetuning.ipynb")