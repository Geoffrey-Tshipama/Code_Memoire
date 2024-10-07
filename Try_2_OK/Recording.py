import os
import datetime
import cv2


def creer_dossier_date():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists(current_date):
        os.makedirs(current_date)
    return current_date


def enregistrer_video(frame, dossier, enregistrement_actif, video_writer):
    if not enregistrement_actif:  # Commencer l'enregistrement
        video_filename = os.path.join(dossier, f'person_detected_{datetime.datetime.now().strftime("%H-%M-%S")}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print(f"Enregistrement commencé : {video_filename}")
        enregistrement_actif = True
    video_writer.write(frame)  # Écrire la frame dans le fichier
    return enregistrement_actif, video_writer