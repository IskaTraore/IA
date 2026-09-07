"""Crée des splits train/valid/test stratifiés (80/10/10) depuis le dataset FieldPlant.

Wrapper fin autour de fieldplant.data.stratified_split — la logique réelle
(regroupement par classe, répartition, réécriture de data.yaml) vit dans
fieldplant/data.py.
"""
import os

from fieldplant.data import DATA_DIR, find_project_root, stratified_split

if __name__ == "__main__":
    root = find_project_root()
    counts = stratified_split(os.path.join(root, DATA_DIR))
    print("Terminé OK :", counts)
