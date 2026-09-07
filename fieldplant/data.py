"""Logique de données partagée : clé API Roboflow, téléchargement, split stratifié.

Ce module est la source de vérité unique. Les scripts (download_dataset.py,
split_dataset.py) et les cellules du notebook y font tous appel, ce qui évite
la duplication et la dérive entre les copies.

Compatible Colab : aucune dépendance à des fichiers locaux du projet, et
find_project_root() retrouve la racine même si le CWD diffère.
"""
from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict

import yaml

# --- Constantes -----------------------------------------------------------
DATA_DIR = "datasets/fieldplant"
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")

SPLITS = {"train": 0.8, "valid": 0.1, "test": 0.1}
SEED = 42
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Nom du workspace/projet Roboflow (FieldPlant, licence CC BY 4.0)
ROBOFLOW_WORKSPACE = "plant-disease-detection"
ROBOFLOW_PROJECT = "fieldplant"


# --- Racine du projet -----------------------------------------------------
def find_project_root() -> str:
    """Retrouve la racine du projet (celle qui contient datasets/ ou .env).

    Compatible Colab : part du CWD et remonte tant que ni datasets/fieldplant
    ni data.yaml du dataset ne sont trouvés. Le CWD fait office de racine en
    dernier recours.
    """
    d = os.getcwd()
    for _ in range(4):
        if os.path.isdir(os.path.join(d, DATA_DIR)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


# --- Clé API --------------------------------------------------------------
def load_roboflow_api_key(project_root: str | None = None) -> str:
    """Retourne la clé API Roboflow depuis l'environnement ou un fichier .env.

    Ordre de priorité :
      1. Variable d'environnement ROBOFLOW_API_KEY
      2. Ligne ROBOFLOW_API_KEY=... dans <project_root>/.env

    Lève SystemExit si aucune clé n'est trouvée.
    """
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if key:
        return key

    root = project_root or find_project_root()
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise SystemExit(
            "ERREUR : clé API Roboflow introuvable. Définis ROBOFLOW_API_KEY "
            "(variable d'environnement) ou dans un fichier .env "
            "(https://app.roboflow.com/settings/api)."
        )
    return key


def dataset_present(project_root: str | None = None) -> bool:
    """True si le dataset (split train) est déjà téléchargé sur disque."""
    root = project_root or find_project_root()
    return os.path.isdir(os.path.join(root, DATA_DIR, "train", "images"))


def download_dataset(project_root: str | None = None, api_key: str | None = None) -> str:
    """Télécharge FieldPlant (format YOLOv8) depuis Roboflow Universe.

    Idempotent : ne fait rien si le dataset est déjà présent.
    Retourne le chemin du dataset.
    """
    root = project_root or find_project_root()
    if dataset_present(root):
        print(f"Dataset déjà présent : {DATA_DIR}. Téléchargement ignoré.")
        return os.path.join(root, DATA_DIR)

    key = api_key or load_roboflow_api_key(root)
    print(f"Clé détectée : {key[:6]}...{key[-4:]} ({len(key)} caractères)")

    from roboflow import Roboflow  # import tardif : la clé est validée avant

    rf = Roboflow(api_key=key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.versions()[0]
    print(f"Version du dataset : {version.version} — téléchargement (format yolov8)...")
    dataset = version.download("yolov8", location=os.path.join(root, DATA_DIR))
    print("Téléchargement terminé.")
    return dataset.location


# --- Split stratifié ------------------------------------------------------
def _group_images_by_class(src: str) -> dict[int, list[str]]:
    """Regroupe les stems d'images par classe principale (1re ligne du label)."""
    by_class: dict[int, list[str]] = defaultdict(list)
    img_dir = os.path.join(src, "images")
    for fname in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue
        label_path = os.path.join(src, "labels", stem + ".txt")
        if not os.path.exists(label_path):
            print(f"⚠️  Label manquant pour {fname} — ignoré")
            continue
        with open(label_path, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        primary = int(lines[0].split()[0]) if lines else -1
        by_class[primary].append(stem)
    return dict(by_class)


def stratified_split(base_dir: str, seed: int = SEED) -> dict[str, int]:
    """Découpe train/ en splits stratifiés 80/10/10 (défaut) et réécrit data.yaml.

    Stratification sur la classe principale de chaque image (1re ligne du
    fichier de label). Idempotent : si valid/ existe déjà, l'opération est
    sautée. Retourne le nombre d'images par split.
    """
    if os.path.isdir(os.path.join(base_dir, "valid", "images")):
        print("Splits déjà créés — étape ignorée.")
        return {
            s: len(os.listdir(os.path.join(base_dir, s, "images")))
            for s in SPLITS
        }

    src = os.path.join(base_dir, "train")
    rng = random.Random(seed)

    by_class = _group_images_by_class(src)
    print(f"Images groupées : {sum(len(v) for v in by_class.values())} "
          f"réparties en {len(by_class)} classes")

    # Répartition par classe : n_train = round(0.8*n), n_valid = round(0.1*n)
    p_train, p_valid = SPLITS["train"], SPLITS["valid"]
    splits: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for cls, stems in sorted(by_class.items()):
        rng.shuffle(stems)
        n = len(stems)
        n_train = round(n * p_train)
        n_valid = round(n * p_valid)
        splits["train"] += stems[:n_train]
        splits["valid"] += stems[n_train:n_train + n_valid]
        splits["test"] += stems[n_train + n_valid:]

    print(f"Répartition : train={len(splits['train'])} "
          f"valid={len(splits['valid'])} test={len(splits['test'])}")

    for split in splits:
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "labels"), exist_ok=True)

    for split, stems in splits.items():
        for stem in stems:
            src_img = next(
                os.path.join(src, "images", stem + e)
                for e in IMAGE_EXTS
                if os.path.exists(os.path.join(src, "images", stem + e))
            )
            shutil.move(src_img, os.path.join(base_dir, split, "images",
                                              os.path.basename(src_img)))
            shutil.move(os.path.join(src, "labels", stem + ".txt"),
                        os.path.join(base_dir, split, "labels", stem + ".txt"))

    rewrite_data_yaml(base_dir)
    return {k: len(v) for k, v in splits.items()}


def rewrite_data_yaml(base_dir: str) -> None:
    """Réécrit data.yaml : chemins absolus + splits train/valid/test.

    Les noms de classes sont lus depuis le yaml existant (aucune liste codée
    en dur). Un fallback minimal préserve nc/names si le fichier manquait.
    """
    yaml_path = os.path.join(base_dir, "data.yaml")
    data: dict = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data.setdefault("nc", 27)
    if "names" not in data:
        raise SystemExit(f"ERREUR : 'names' absent de {yaml_path} — "
                         "impossible de réécrire data.yaml sans les classes.")

    data["path"] = os.path.abspath(base_dir)
    data["train"], data["val"], data["test"] = "train/images", "valid/images", "test/images"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# FieldPlant (Roboflow Universe) — splits créés localement "
                f"(80/10/10, seed {SEED})\n")
        f.write("# Source : https://universe.roboflow.com/"
                f"{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}\n")
        yaml.dump(data, f, allow_unicode=True)
    print(f"data.yaml réécrit : {yaml_path}")
