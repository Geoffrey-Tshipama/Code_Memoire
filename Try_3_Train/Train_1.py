"""
But : Nous allons entrainer notre modèle YOLOv8 et exporter ce modèle

Évaluation 1 : training avec un dataset contenant des images de guns, ketch...

"""

from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        patience=25,
        batch=-1,
        imgsz=640,
        cache=True,
        device=0,
        lr0=0.0001,
        lrf=0.5,
        plots=True
    )


