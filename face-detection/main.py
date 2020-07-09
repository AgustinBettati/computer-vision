import sys
import face_recognition
import cv2
import imutils as imutils
import numpy as np
import yaml
from mosaic_view import mosaic_view, concat_face_imgs
from person import Person

# TODO:
# - mostrar imagen de perfil y una del video

video_path = "/Users/abettati/Desktop/videos-vision/Screen Recording 2020-07-02 at 8.34.21 PM.mov"
# video_path = "/Users/abettati/Desktop/grande-detecta-bien.mov"
# video_path = "/Users/abettati/Desktop/no-detecta-nada.mov"
video_capture = cv2.VideoCapture(video_path)

# Load a sample picture and learn how to recognize it.

participants = []
with open('./input/alumnos.yaml') as f:
  data = yaml.load(f, Loader=yaml.FullLoader)
  for student_info in data:
    participants.append(Person(student_info[0], student_info[1]))


def from_file_to_encoding(filename):
  image = face_recognition.load_image_file(filename)
  face_encoding = face_recognition.face_encodings(image)[0] # ideally only one face is detected
  return face_encoding

for person in participants:
  encoding = from_file_to_encoding(person.picture_url)
  person.img = cv2.imread(person.picture_url)
  person.encoding = encoding

known_face_encodings = list(map(lambda p: p.encoding, participants))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while (video_capture.isOpened):
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if not frame is None:

        h, w = frame.shape[:2]
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = imutils.resize(frame, width=1000)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            # face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0, model="cnn")
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for i in range(len(face_encodings)):
                face_encoding = face_encodings[i]
                face_location = face_locations[i]
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.50)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    participant = participants[best_match_index]
                    participant.is_present = True
                    name = participant.name
                    (top, right, bottom, left) = face_locations[i]
                    margin = 35
                    video_image = small_frame[top -margin: bottom + margin, left - margin:right + margin]
                    if len(participant.video_img) == 0:
                      participant.video_img = video_image.copy()

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):

            # Draw a box around the face
            margin = 20
            cv2.rectangle(small_frame, (left - margin, top - margin), (right + margin, bottom + margin), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(small_frame, (left - margin, bottom + margin - 20), (right + margin, bottom + margin), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(small_frame, name, (left - margin + 6, bottom + margin - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', small_frame)
    else:
      break
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

present = list(filter(lambda p: p.is_present, participants))
absent = list(filter(lambda p: not p.is_present, participants))

present_img = mosaic_view(list(map(lambda p: (concat_face_imgs(p), p), present)))
absent_img = mosaic_view(list(map(lambda p: (p.img, p), absent)))
if not present_img is None:
  cv2.imwrite('./output/present.jpg', present_img)
if not absent_img is None:
  cv2.imwrite('./output/absent.jpg', absent_img)

# Release handle to the webcam
with open('./output/asistencia.yaml', 'w') as outfile:
  result = {
    "presentes": list(map(lambda p: p.name, present)),
    "ausentes": list(map(lambda p: p.name, absent))
  }
  yaml.dump(result, outfile, explicit_start=True, default_flow_style=False)
video_capture.release()
cv2.destroyAllWindows()
sys.exit()
