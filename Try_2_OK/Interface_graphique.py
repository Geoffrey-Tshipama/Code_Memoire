import cv2
from ultralytics import YOLO
from screeninfo import get_monitors
import pygame


# Charger le modèle YOLOv8
model = YOLO('yolov8m.pt')  # Vous pouvez choisir un autre modèle YOLOv8 selon vos besoins

CONFIDENCE_THRESHOLD = 0.7  # C'est prévue une conf = 0,8

# Remplacez 'rtsp://username:password@ip_address:port/stream' par l'URL RTSP de votre caméra
rtsp_url = 'rtsp://admin:Burotop01@10.10.50.127:554/Streaming/Channels/101'

# Initialiser pygame pour l'audio
pygame.mixer.init()

# Charger le fichier audio
sound = pygame.mixer.Sound('son_Alarm.wav')

# Créez un objet VideoCapture
cap = cv2.VideoCapture(rtsp_url)

# Vérifiez si la connexion est ouverte
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le flux RTSP")
else:
    print("Connexion réussie au flux RTSP")

# Lire et afficher les images
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo")
        break

    # Effectuer la détection sur l'image
    results = model(frame)

    person_detected = False  # Indicateur pour la détection de personnes

    # Filtrer uniquement les personnes (class_id = 0 pour la classe 'person' dans les modèles YOLO)
    for result in results:
        boxes = result.boxes  # Obtenir les boîtes englobantes
        for box in boxes:
            class_id = int(box.cls)

            # Supprimer les détections en dessous du seuil CONFIDENCE_THRESHOLD
            if float(box.conf) < CONFIDENCE_THRESHOLD:
                continue

            # Détection que de la classe personne
            if class_id == 0:  # 0 correspond à la classe 'person'
                # Récupérer les coordonnées de la boîte englobante
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Dessiner la boîte englobante et l'étiquette
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Définir que la personne est détectée
            person_detected = True

    if person_detected:
        sound.play()

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
