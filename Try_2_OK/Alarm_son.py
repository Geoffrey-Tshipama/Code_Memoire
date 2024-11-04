import cv2
from ultralytics import YOLO
import pygame
import time

# Charger le modèle YOLOv8
model = YOLO('yolov8n.pt')

# Initialiser pygame pour l'audio
pygame.mixer.init()

# URL RTSP de la caméra (remplacez par l'URL de votre caméra)
rtsp_url = "rtsp://admin:Burotop01@10.10.50.114:554/stream"

# Charger le fichier audio
sound = pygame.mixer.Sound('son_Alarm.wav')

# Charger la vidéo ou accéder au flux de la caméra
video_path = 'Entrée/2.mp4'  # Remplacez par le chemin de votre vidéo
video = cv2.VideoCapture(video_path)

while True:
    # Lire les images de la vidéo
    ret, frame = video.read()
    if not ret:
        break

    # Dans la boucle
#   time.sleep(0.1)  # Attendre 100ms entre les itérations

    # Effectuer la détection avec YOLOv8
    results = model(frame)

    # Vérifier les détections
    for result in results:
        for bbox in result.boxes:
            # Obtenir la classe détectée (index 0 correspond à "personne")
            class_id = int(bbox.cls)
            if class_id == 0:  # 0 est l'ID de la classe "person" dans le modèle COCO
                # Émettre le son
                sound.play()

    # Afficher l'image avec les détections
    cv2.imshow('YOLOv8 Detection', frame)

    # Quitter la boucle en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video.release()
cv2.destroyAllWindows()

pygame.mixer.quit()
