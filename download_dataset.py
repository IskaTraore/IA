"""Téléchargement du dataset FieldPlant (Roboflow Universe) au format YOLOv8.

Wrapper fin autour de fieldplant.data.download_dataset — la logique réelle
(clé API, idempotence) vit dans fieldplant/data.py.
"""
from fieldplant.data import download_dataset

if __name__ == "__main__":
    download_dataset()
