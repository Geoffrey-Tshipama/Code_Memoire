import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from screeninfo import get_monitors
import pygame
import Recording

# Charger le modèle YOLOv8
model = YOLO('best_14.pt')  # Vous pouvez choisir un autre modèle YOLOv8 selon vos besoins

CONFIDENCE_THRESHOLD = 0.5
# C'est prévue une conf = 0,8
EXCLUDED_CLASS_ID = 0  # L'exclusion de la classe 0 qui détecte de personne non violente

# Remplacez 'rtsp://username:password@ip_address:port/stream' par l'URL RTSP de votre caméra
rtsp_url = 'rtsp://admin:Burotop01@10.10.50.115:554/Streaming/Channels/101'

# Initialiser pygame pour l'audio
pygame.mixer.init()

# Charger le fichier audio
sound = pygame.mixer.music.load('son_Alarm.wav')

# Récupération des noms des classes
class_names = model.names  # Cela vous donne un dictionnaire avec les IDs comme clés et les noms comme valeurs

print(class_names)

# Créez un objet VideoCapture
cap = cv2.VideoCapture(rtsp_url)

tracker = DeepSort(max_age=50)

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

    detected = False

    # Effectuer la détection sur l'image
    detections = model(frame)[0]

    results = []

    #    person_detected = False  # Indicateur pour la détection de personnes

    # Filtrer uniquement les personnes (class_id = 0 pour la classe 'person' dans les modèles YOLO)
    for data in detections.boxes.data.tolist():
        # La probabilité de chaque détection
        confidence = data[4]

        # Récupérez l'id du nom de la classe
        class_id = int(data[5])

        # Supprimer les détections en dessous du seuil CONFIDENCE_THRESHOLD
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        if class_id == EXCLUDED_CLASS_ID:
            continue
        # Détection que de la classe personne
        # if class_id != 0:  # 0 correspond à la classe 'person'
        # Récupérer les coordonnées de la boîte englobante
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # Dessiner la boîte englobante et l'étiquette
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Définir que la personne est détectée
        # person_detected = True

        # ajouter le cadre de délimitation (x, y, l, h), la confiance et l'identifiant de classe à la liste des
        # résultats
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        class_id = track.get_det_class()  # Get the class ID from the track
        class_name = class_names[class_id]  # Get the class name from the class ID

        # Jouer un son si la classe "violence" est détectée
        if class_name.lower() != "NonViolence":
            detected = True
        # pygame.mixer.music.play()

        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Draw the class name above the bounding box
        cv2.putText(frame, f"{class_name} ({track_id})", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), (0, 255, 0), -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  #

    if ret:
        pygame.mixer.music.play()
        recording, out = Recording.enregistrer_video(frame, dossier_create, recording, out)
    elif recording:
        pygame.mixer.stop()
        recording = False
        out.release()
        print("Fin de l'enregistrement")

    """
    if person_detected:
        sound.play()
        recording, out = Recording.enregistrer_video(frame, dossier_create, recording, out)
    elif recording:  # Arrêter l'enregistrement si aucune personne n'est détectée
        pygame.mixer.stop() # Arret du son
        recording = False
        out.release()
        print("Enregistrement arrêté")
    """

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
