import cv2
import os


def extract_images_from_video(video_path, output_folder):
    # Ouvre la vidéo
    cap = cv2.VideoCapture(video_path)

    # Vérifie si la vidéo est ouverte correctement
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo")
        return

    # Crée le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Récupère le taux de frames par seconde (FPS) de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialisation du compteur d'images
    frame_number = 0

    while True:
        # Lit une image
        ret, frame = cap.read()

        # Si la vidéo est terminée, on sort de la boucle
        if not ret:
            break

        # Si la frame est une image à enregistrer (chaque seconde)
        if frame_number % int(fps) == 0:
            # Crée le chemin de sauvegarde pour l'image
            image_filename = os.path.join(output_folder, f"image_{frame_number // int(fps)}.jpg")

            # Enregistre l'image
            cv2.imwrite(image_filename, frame)
            print(f"Image sauvegardée : {image_filename}")

        # Incrémente le numéro de la frame
        frame_number += 1

    # Libère la vidéo
    cap.release()
    print("Extraction terminée.")


# Exemple d'utilisation
video_path = "bagarre_20-30-41.mp4.avi"  # Remplacez par le chemin de votre vidéo
output_folder = "images_extraite_2"  # Dossier où les images seront enregistrées

extract_images_from_video(video_path, output_folder)
