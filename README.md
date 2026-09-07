# Détection de maladies foliaires — YOLOv8 & FieldPlant

Fine-tuning de **YOLOv8n** pour localiser (bounding box) et classer **27 maladies foliaires**
(manioc, maïs, tomate) sur des photos de terrain, avec le dataset
[FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant)
(5 156 images annotées, CC BY 4.0).

> Citation : Moupojou, E., et al. *FieldPlant: A Dataset of Field Plant Images for Plant
> Disease Detection and Classification With Deep Learning.* IEEE Access, vol. 11, 2023.

---

## 🚀 Entraîner le modèle sur Google Colab (recommandé)

L'entraînement complet (30 époques, 640 px, 5 156 images) prend environ **30 minutes sur le
GPU gratuit de Colab**, contre plusieurs jours sur un CPU local.

### Ouvrir le notebook dans Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IskaTraore/IA/blob/main/notebook_finetuning.ipynb)

> Le badge ouvre le notebook directement dans Colab depuis ce dépôt.
>
> **Alternative sans dépôt** : allez sur [colab.research.google.com](https://colab.research.google.com/)
> → `Fichier` → `Importer un notebook` → uploadez `notebook_finetuning.ipynb`.

### Étapes d'entraînement

1. **Activer le GPU** — `Exécution` → `Modifier le type d'exécution` → **T4 GPU** → Enregistrer.
2. **Exécuter la cellule 1** (installation) : installe `ultralytics` et récupère le module
   partagé `fieldplant/` depuis le dépôt.
   - ⚠️ Si le téléchargement du module échoue, copiez manuellement le dossier `fieldplant/`
     du projet dans le répertoire du notebook (`Fichier` → panneau latéral → upload).
3. **Obtenir une clé API Roboflow** (gratuite) :
   - Créez un compte sur [app.roboflow.com](https://app.roboflow.com)
   - `Settings` → `API Keys` → copiez la clé privée.
4. **Exécuter la cellule 2** (téléchargement du dataset) : collez la clé quand le champ
   s'affiche (ou définissez `ROBOFLOW_API_KEY` dans les secrets Colab : icône 🔑 à gauche).
   → Le dataset FieldPlant (5 156 images) est téléchargé au format YOLOv8.
5. **Exécuter la cellule 3** (split) : découpage stratifié **80/10/10** (seed 42)
   → ~4 128 train / 515 valid / 513 test.
6. **Cellules 4 (exploration)** : distribution des 27 classes + exemples annotés.
7. **Cellule 5 (entraînement)** : fine-tuning YOLOv8n (~1 min/époque sur T4, early
   stopping patience 15). **Reprise automatique** : si la session est interrompue,
   relancez simplement la cellule — elle repart depuis `last.pt`.
8. **Cellules 6-8 (évaluation)** : métriques sur le split test (mAP@0.5, mAP@0.5:0.95,
   précision, rappel, F1 par classe), matrices de confusion, courbes PR, exemples
   qualitatifs et benchmark d'inférence.
9. **Récupérer les résultats** (le disque Colab est éphémère !) :
   - `runs/detect/yolo_gpu/weights/best.pt` → **les poids du modèle entraîné**
   - `figures/` → graphiques et CSV des métriques
   - Panneau latéral `Fichiers` → clic droit → `Télécharger`
   - Option durable : monter Google Drive (`Fichiers` → icône Drive) avant l'entraînement
     et copier `best.pt` dedans.

---

## 💻 Utilisation en local (CPU)

Préparer l'environnement, puis valider le pipeline (l'entraînement complet sur CPU n'est
pas réaliste — voir section Colab ci-dessus) :

```bash
# 1. Environnement virtuel + dépendances
python -m venv .venv
.venv/Scripts/activate            # Windows (bash : source .venv/Scripts/activate)
pip install ultralytics roboflow nbformat nbclient pyyaml pandas matplotlib

# 2. Clé API Roboflow
cp .env.example .env              # puis renseigner ROBOFLOW_API_KEY=...

# 3. Télécharger le dataset (idempotent)
python download_dataset.py

# 4. Split stratifié 80/10/10 (idempotent)
python split_dataset.py

# 5. Smoke test : 40 images, 1 époque (~2 min) — valide tout le pipeline
python train.py --smoke
```

### Exécuter le notebook en local (mode CPU réduit)

```bash
python run_notebook.py
```

Exécute le notebook sans interface (nbclient) en mode `cpu_subset`
(320 px, 20 époques, 12 % des données). En cas de timeout cellule (480 s), la progression
est sauvegardée — relancez simplement la commande pour continuer.

---

## 📁 Structure du projet

| Fichier | Rôle |
|---|---|
| `notebook_finetuning.ipynb` | **Livrable final** — pipeline complet autonome (Colab GPU ou CPU local) |
| `build_notebook.py` | Génère le notebook (éditer la logique ici, pas dans le `.ipynb`) |
| `run_notebook.py` | Exécute le notebook headless en local (mode CPU réduit) |
| `fieldplant/data.py` | **Module partagé** — clé API, téléchargement Roboflow, split stratifié |
| `download_dataset.py` | Wrapper : téléchargement du dataset |
| `split_dataset.py` | Wrapper : split train/valid/test 80/10/10 |
| `train.py` | Entraînement CPU local + mode `--smoke` |
| `datasets/fieldplant/` | Dataset (généré, non versionné) |
| `figures/`, `runs/` | Figures et résultats d'entraînement (générés, non versionnés) |

Après toute modification de `build_notebook.py`, régénérer le notebook :

```bash
python build_notebook.py
```
