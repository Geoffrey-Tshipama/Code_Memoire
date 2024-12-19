"""
But : Nous allons entrainer notre modèle YOLOv8 et exporter ce modèle

Évaluation 1 : training avec un dataset contenant des images de guns, ketch...

Evalution -1 : Nous voici presque à la fin de notre programme, maintenant, nous entrainons notre modèle
                avec un dataset qu'on a rassemblé nous meme en prenant certains données au sain de la faculté
                polytechnique.

"""

from ultralytics import YOLO
import os

if __name__ == '__m   ain__':
    model = YOLO("yolov8s.pt")

    # Resume training
    # results = model.train(resume=True)

    results = model.train(
        data="data.yaml",
        epochs=100,
        patience=25,
        batch=-1,
        imgsz=640,
        cache=True,
        device=0,
        lr0=0.0001,
        lrf=0.1,
        plots=True)





