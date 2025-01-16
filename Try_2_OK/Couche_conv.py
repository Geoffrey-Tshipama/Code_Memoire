import torch
from ultralytics import YOLO  # Si vous utilisez YOLOv8 directement

# Charger un modèle YOLOv8s
model = YOLO('best_5.pt')

# Afficher la structure
print(model.model)

# Fonction pour compter les convolutions
def count_conv_layers(model):
    conv_count = 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_count += 1
    return conv_count

# Compter les convolutions
conv_layers = count_conv_layers(model.model)
print(f"Nombre total de couches de convolution : {conv_layers}")
