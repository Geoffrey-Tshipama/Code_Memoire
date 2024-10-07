import cv2
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO('yolov8n.pt')  # Vous pouvez choisir un autre modèle YOLOv8 selon vos besoins

# Remplacez 'rtsp://username:password@ip_address:port/stream' par l'URL RTSP de votre caméra
rtsp_url = 'rtsp://admin:Burotop01@10.10.50.127:554/Streaming/Channels/101'

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

    # Filtrer uniquement les personnes (class_id = 0 pour la classe 'person' dans les modèles YOLO)
    for result in results:
        boxes = result.boxes  # Obtenir les boîtes englobantes
        for box in boxes:
            class_id = int(box.cls)
            if class_id == 0:  # 0 correspond à la classe 'person'
                # Récupérer les coordonnées de la boîte englobante
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Dessiner la boîte englobante et l'étiquette
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Redimensionner l'image (par exemple, à 50% de la taille originale)
    scale_percent = 70  # Pourcentage de redimensionnement
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Afficher la vidéo
    window_name = 'RTSP Stream'
    cv2.imshow(window_name, resized_frame)

    # Obtenir les dimensions de l'écran
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Calculer la position pour centrer la fenêtre
    x = int((screen_width - width) / 2)
    y = int((screen_height - height) / 2)

    # Placer la fenêtre au centre de l'écran
    cv2.moveWindow(window_name, x, y)

    # Sortir de la boucle avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer l'objet VideoCapture et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
