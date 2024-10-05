import datetime
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from helper import create_video_writer
import pygame  # Importer pygame pour jouer le son

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Initialiser pygame et charger le son
pygame.mixer.init()
sound = pygame.mixer.music.load('son.wav')  # Remplacer par le chemin du fichier son

# initialize the video capture object
video_cap = cv2.VideoCapture("Entrée/Fight/Baston.mp4")
# initialize the video writer object
writer = create_video_writer(video_cap, "Sortie_with_DeepSORT/output_Baston")

# load the pre-trained YOLOv8n model
model = YOLO("best.pt")

# Récupération des noms des classes
class_names = model.names  # Cela vous donne un dictionnaire avec les IDs comme clés et les noms comme valeurs


# The max_age parameter in DeepSort specifies the maximum number of missed frames before an unallocated track is
# discarded. In this example it is set to 50, which means that if an object is not detected for 50 frames its track
# will be removed
tracker = DeepSort(max_age=10)


while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # Obtenir le nom de la classe à partir de l'ID
        # class_name = class_names[class_id]
        # Cela vous donne le nom de la classe

        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    ######################################
    # TRACKING
    ######################################

    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        class_id = track.get_det_class()  # Get the class ID from the track
        class_name = class_names[class_id]  # Get the class name from the class ID

        # Jouer un son si la classe "violence" est détectée
        if class_name.lower() == "Violence":
            pygame.mixer.music.play()

        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        # Draw the class name above the bounding box
        cv2.putText(frame, f"{class_name} ({track_id})", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)  #

    # heure de fin pour calculer les fps
    end = datetime.datetime.now()
    # Afficher le temps qu'il a fallu pour traiter 1 image
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # afficher le cadre sur notre écran
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()
