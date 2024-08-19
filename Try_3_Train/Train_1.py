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
        epochs=10,
        imgsz=640,
        device=0
    )

    save_dir = r"C:\Users\Fablabpolytech\PycharmProjects\Code_GTM\Memoire\Try_3_Train\Trained"
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, "trained_model_1"))
