import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from screeninfo import get_monitors
import pygame
import Recording

# Charger le modèle YOLOv8
model = YOLO('best_5.pt')

CONFIDENCE_THRESHOLD = 0.6  # Le seuil de détection
# EXCLUDED_CLASS_ID = 3
# L'exclusion de la classe 3 qui détecte de personne non violente

# L'URL RTSP de ma caméra
rtsp_url = 'rtsp://admin:Burotop01@10.10.50.122:554/Streaming/Channels/101'

# Initialiser pygame pour l'audio
pygame.mixer.init()

# Charger le fichier audio
pygame.mixer.music.load('son_Alarm.wav')

# Récupération des noms des classes
class_names = model.names
# Cela donne un dictionnaire avec les IDs comme clés et les noms comme valeurs

# Créez un objet VideoCapture
cap = cv2.VideoCapture(rtsp_url)

tracker = DeepSort(max_age=30)
# max_age est le nombre maximum des frames de tenir une prédiction avant de le faire disparaitre


# Vérifiez si la connexion est ouverte
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le flux RTSP")
else:
    print("Connexion réussie au flux RTSP")

# Créer le dossier pour contenir les enrégistrements du jour
dossier_create = Recording.creer_dossier_date()

# Variables pour l'enregistrement vidéo
recording = False
out = None

# Lire et afficher les images
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo")
        break

    # Effectuer la détection sur l'image
    detections = model(frame)[0]

    results = []

    detected = False  # Initialiser la variable de détection

    for data in detections.boxes.data.tolist():
        # La probabilité de chaque détection
        confidence = data[4]

        # Récupérez l'id du nom de la classe
        class_id = int(data[5])

        # Supprimer les détections en dessous du seuil CONFIDENCE_THRESHOLD
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if class_id == EXCLUDED_CLASS_ID:
        # continue

        # Récupérer les coordonnées de la boîte englobante
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        # ajouter le cadre de délimitation (x, y, l, h), la confiance et l'identifiant de classe à la liste des
        # résultats
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        detected = True  # Activer le flag si une détection est confirmée

        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        class_id = track.get_det_class()  # Obtenir l'ID de la classe depuis la piste.
        class_name = class_names[class_id]  # Récupérez le nom de la classe

        # Jouer un son si la classe "violence" est détectée
        # if class_name.lower() != "NonViolence":
        # detected = True
        # pygame.mixer.music.play()

        # dessiner la boite englobante, mettre id du track et le nom de la classe au dessus de la boite
        # Draw the class name above the bounding box
        cv2.putText(frame, f"{class_name} ({track_id})", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), (0, 255, 0), -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  #

    if detected:
        pygame.mixer.music.play()
        recording, out = Recording.enregistrer_video(frame, dossier_create, recording, out)
    elif recording:
        pygame.mixer.stop()
        recording = False
        out.release()
        print("Fin de l'enregistrement")

    # Redimensionner l'image (par exemple, à 50% de la taille originale)
    scale_percent = 50  # Pourcentage de redimensionnement
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Afficher la vidéo
    window_name = 'RTSP Stream'
    cv2.imshow(window_name, resized_frame)

    # Obtenir les dimensions de l'écran
    # screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Obtenir les dimensions de l'écran principal
    screen = get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height

    # Calculer la position pour centrer la fenêtre
    x = int((screen_width - width) / 2)
    y = int((screen_height - height) / 2)

    # Placer la fenêtre au centre de l'écran
    cv2.moveWindow(window_name, x, y)

    # Vérifier les entrées clavier
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quitter avec la touche 'q'
        break
    elif key == ord(' '):  # Appuyer sur la touche espace pour couper/activer le son
        pygame.mixer.stop()  # Arrêter tous les sons

# Libérer l'objet VideoCapture et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
