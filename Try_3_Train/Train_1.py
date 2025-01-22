"""
But : Nous allons entrainer notre modèle YOLOv8s et exporter ce modèle pour faire l'inférence.

Évaluation : training avec un dataset contenant des images de Guns, Non_Violence, Violence, Knife.
             Ce code génère un ensemble des dossiers utiles qui servent à suivre et évaluer le
             modèle pour voir les performances de ce dernier.

             Pour mieux comprendre, nous pouvons définir quelques hyper-paramètres présentés :
             - cache = true : permet de mettre les élements du dataset dans la RAM pour éviter
                              d'aller chercher les images à chaque fois dans le disque dur.
             - batch = -1 : choisi de manière automatique les nombres d'éléments du dataset à
                            traiter de manière continue avant d'appliquer une retro-progression
                            sur l'ensemble du modèle.
             - lr = 0.0001 : est le taux d'apprentissage appliqué lors de l'apprentissage du modèle
                             pour réduire l'erreur, la fonction de perte.

"""

from ultralytics import YOLO
import os

if __name__ == '__main__':
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





