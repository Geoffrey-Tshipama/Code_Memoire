"""
Ce code lit une vidéo, détecte des objets dans chaque frame à l'aide d'un modèle YOLO, dessine des rectangles autour des
objets détectés, affiche le nombre d'images par seconde, et enregistre le résultat dans un nouveau fichier vidéo.
"""
import datetime
import cv2
from ultralytics import YOLO
from helper import create_video_writer

# define some constants
CONFIDENCE_THRESHOLD = 0.8  # C'est prévue une conf = 0,8
GREEN = (0, 255, 0)
# EXCLUDED_CLASS_ID = 2

# initialize the video capture object
video_cap = cv2.VideoCapture("Entrée/Fac_2_.mp4")
# initialize the video writer object
writer = create_video_writer(video_cap, "Sortie_without_DeepSORT/Train3/output_fac_2_.mp4")

# load the pre-trained YOLOv8n model
model = YOLO("best.pt")


while True:
    # start time to compute the fps
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # extract class ID
        # class_id = int(data[5])

        # check if the detected class is the one to exclude
        # if class_id == EXCLUDED_CLASS_ID:
            # continue  # skip this detection if it's the excluded class

        # if the confidence is greater than the minimum confidence,
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        # draw the bounding box on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)

        # get the class name using the class ID
        class_name = model.names[class_id]

        # draw the class name above the bounding box
        cv2.putText(frame, f"{class_name} ({confidence:.2f})",
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()
