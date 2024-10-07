import os
import cv2
import datetime
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO('yolov8n.pt')

# URL RTSP de la caméra (remplacez par l'URL de votre caméra)
rtsp_url = "rtsp://admin:Burotop01@10.10.50.127:554/Streaming/Channels/101"

# Méthode pour créer un dossier avec la date du jour
def creer_dossier_date():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists(current_date):
        os.makedirs(current_date)
    return current_date

# Méthode pour démarrer l'enregistrement vidéo
def enregistrer_video(frame, dossier, enregistrement_actif, video_writer):
    if not enregistrement_actif:  # Commencer l'enregistrement
        video_filename = os.path.join(dossier, f'person_detected_{datetime.datetime.now().strftime("%H-%M-%S")}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print(f"Enregistrement commencé : {video_filename}")
        enregistrement_actif = True
    video_writer.write(frame)  # Écrire la frame dans le fichier
    return enregistrement_actif, video_writer

# Ouvrir le flux RTSP
video_capture = cv2.VideoCapture(rtsp_url)

# Vérifier si la connexion au flux est établie
if not video_capture.isOpened():
    print("Erreur : Impossible de se connecter au flux RTSP")
    exit()

# Créer un dossier avec la date du jour
dossier_date = creer_dossier_date()

# Variables pour l'enregistrement vidéo
recording = False
out = None

while True:
    # Lire une image du flux
    ret, frame = video_capture.read()
    if not ret:
        print("Erreur lors de la lecture du flux RTSP")
        break

    # Effectuer la détection avec YOLOv8
    results = model(frame)

    # Vérifier s'il y a des personnes détectées
    person_detected = False
    for result in results:
        for bbox in result.boxes:
            # Obtenir la classe détectée (index 0 correspond à "personne")
            class_id = int(bbox.cls)
            if class_id == 0:  # 0 est l'ID de la classe "person" dans le modèle COCO
                person_detected = True
                # Afficher la boîte englobante sur l'image
                x1, y1, x2, y2 = bbox.xyxy[0].tolist()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Gérer l'enregistrement vidéo
    if person_detected:
        recording, out = enregistrer_video(frame, dossier_date, recording, out)
    elif recording:  # Arrêter l'enregistrement si aucune personne n'est détectée
        recording = False
        out.release()
        print("Enregistrement arrêté")

    # Afficher l'image avec les détections
    cv2.imshow('RTSP Camera Stream - YOLOv8 Detection', frame)

    # Quitter la boucle en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
if recording:
    out.release()
video_capture.release()
cv2.destroyAllWindows()
