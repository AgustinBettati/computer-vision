import sys
import face_recognition
import cv2
import imutils as imutils
import numpy as np
import yaml
from mosaic_view import mosaic_view, concat_face_imgs
from person import Person

video_path = "/Users/abettati/Desktop/resultado/4-alumnos.mov"

video_capture = cv2.VideoCapture(video_path)

# cuanto mas chica es la imagen que se usa para procesar
scale = 4

participants = []
with open('./input/alumnos.yaml') as f:
  data = yaml.load(f, Loader=yaml.FullLoader)
  for student_info in data:
    participants.append(Person(student_info[0], student_info[1]))


def from_file_to_encoding(filename):
  image = face_recognition.load_image_file(filename)
  locations = face_recognition.face_locations(image)
  face_encoding = face_recognition.face_encodings(image, locations)[0]
  return face_encoding

for person in participants:
  encoding = from_file_to_encoding(person.picture_url)
  person.img = cv2.imread(person.picture_url)
  person.encoding = encoding

known_face_encodings = list(map(lambda p: p.encoding, participants))

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while (video_capture.isOpened):
    ret, original_frame = video_capture.read()

    if not original_frame is None:

        frame = imutils.resize(original_frame, width=1400)
        small_frame = cv2.resize(frame, (0, 0), fx= 1/scale, fy= 1/scale)

        rgb_small_frame = small_frame[:, :, ::-1]

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
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.52)

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    # la cara mas cercana y esta dentro de la tolerancia
                    participant = participants[best_match_index]
                    participant.is_present = True
                    name = participant.name
                    (top, right, bottom, left) = list(map(lambda x: x * scale, face_locations[i]))
                    margin = 35
                    video_image = frame[top -margin: bottom + margin, left - margin:right + margin]
                    if len(participant.video_img) == 0:
                      participant.video_img = video_image.copy()

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            # Draw a box around the face
            margin = 20
            cv2.rectangle(frame, (left - margin, top - margin), (right + margin, bottom + margin), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left - margin, bottom + margin - 20), (right + margin, bottom + margin), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left - margin + 6, bottom + margin - 6), font, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        # cv2.imshow('Procesamiento', small_frame)
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
