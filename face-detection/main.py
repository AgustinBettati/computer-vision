import glob
import sys

import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
import yaml

from mosaic_view import mosaic_view
from person import Person

# TODO:
# - mostrar imagen de perfil y una del video

video_path = "/Users/abettati/Desktop/Screen Recording 2020-07-02 at 8.34.21 PM.mov"
video_capture = cv2.VideoCapture(video_path)

# Load a sample picture and learn how to recognize it.

participants = []
with open('./alumnos.yaml') as f:
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
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.6)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    participants[best_match_index].is_present = True
                    name = participants[best_match_index].name

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
    else:
      break
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

present = list(filter(lambda p: p.is_present, participants))
absent = list(filter(lambda p: not p.is_present, participants))

cv2.imwrite('./present.jpg', mosaic_view(list(map(lambda p: p.img, present))))
cv2.imwrite('./absent.jpg', mosaic_view(list(map(lambda p: p.img, absent))))

# Release handle to the webcam
with open('asistencia.yaml', 'w') as outfile:
  result = {
    "presentes": list(map(lambda p: p.name, present)),
    "ausentes": list(map(lambda p: p.name, absent))
  }
  yaml.dump(result, outfile, explicit_start=True, default_flow_style=False)
video_capture.release()
cv2.destroyAllWindows()
sys.exit()
