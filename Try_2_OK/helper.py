"""
Cette fonction permet de configurer un écrivain vidéo pour capturer des images d'une source vidéo et les enregistrer
dans un fichier au format spécifié.

Un écrivain vidéo, ou VideoWriter, est un outil qui permet de créer et d'enregistrer des vidéos à partir d'images
(ou de frames).

En d'autres termes, c'est comme une machine qui prend des photos à une certaine vitesse (images par seconde) et les
assemble pour créer un film. Vous lui dites où sauvegarder le film, quel format utiliser, et il s'occupe du reste !
"""
import cv2


def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # J'ai le code MP4V en minuscule
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer
