import math
import cv2
from drawing_utils import mosaic_view
from vehicle import Vehicle

video_path = 'carsRt9_3.avi'
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
background_subs = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=16)

car_list = []
historic_car_imgs = []

p1 = (0, 190)
p2 = (640, 174)

frame_counter = 0
check_velocity = False

cap = cv2.VideoCapture(video_path)

def obtain_color(area, is_top, cY):
  if (is_top and area < 1300) or (not is_top and cY <= 230 and area < 1500) or (not is_top and cY > 230 and area < 4200):
    return (0, 255, 0)
  if (is_top and area < 2500) or (not is_top and area < 3000):
    return (0, 255, 255)
  else:
    return (0, 0, 255)

def contour_to_img(img, contour):
  (x, y, w, h) = cv2.boundingRect(contour)
  crop_image = img[y:y + h, x:x + w]
  return crop_image

def calculateDistance(x1, y1, x2, y2):
  dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return dist

def obtain_center(contour):
  M = cv2.moments(contour)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  return [cX, cY]

# flip en true implica poner en negro la parte de abajo
def half_black(img, flip):
  for val in enumerate(img):
    (y_index, x_array) = val
    if flip and y_index < 174:
      continue
    if not flip and y_index > 190:
      continue
    for x_val in enumerate(x_array):
      (x_index, _) = x_val
      if flip and y_index > 190:
        x_array[x_index] = 0
      elif not flip and y_index < 174:
        x_array[x_index] = 0
      else:
        y_result = -0.025 * x_index + 190
        if flip and y_index > y_result:
          x_array[x_index] = 0
        if not flip and y_index < y_result:
          x_array[x_index] = 0

def draw_contour(c, frame, velocity, is_top, cY):
  # get bounding box from countour
  (x, y, w, h) = cv2.boundingRect(c)
  # draw bounding box
  cv2.rectangle(frame, (x, y), (x + w, y + h), obtain_color(cv2.contourArea(c), is_top, cY), 2)
  cv2.putText(frame, str(velocity) + " km/h", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def calculate_velocity(cX, cY, car, is_top):
  distance = calculateDistance(cX, cY, car.hist_x, car.hist_y)
  if is_top:
    distanceM = distance * 0.102
  else:
    distanceM = distance * 0.07
  meterPerSecond = distanceM / 0.1
  return int(meterPerSecond * 3.6)

while (cap.isOpened):

  ret, frame = cap.read()
  clean_frame = frame.copy()

  # contador para definir cada cuantos frames se calcula velocidad
  frame_counter = frame_counter + 1
  if (frame_counter > 3):  # cada 0.1s (el video es 30 fps)
    check_velocity = True
    frame_counter = 0

  # apply background substraction
  detected_motion = background_subs.apply(frame)

  # marcar los autos mejor
  detected_motion = cv2.morphologyEx(detected_motion, cv2.MORPH_CLOSE, kernel)  # Dilation followed by Erosion

  # sacar ruido del fondo
  detected_motion = cv2.morphologyEx(detected_motion, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation

  top_half = detected_motion.copy()
  half_black(top_half, True)

  bottom_half = detected_motion.copy()
  half_black(bottom_half, False)

  # return outer contours
  (top_contours, _) = cv2.findContours(top_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  (bottom_contours, _) = cv2.findContours(bottom_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  def valid_countor(c):
    (x, y, width, h) = cv2.boundingRect(c)
    return (cv2.contourArea(c) > 500 and width < 150)

  filtered_contours_top = list(filter(valid_countor, top_contours))
  filtered_contours_bottom = list(filter(valid_countor, bottom_contours))

  car_imgs = list(map(lambda c: contour_to_img(frame, c), filtered_contours_bottom + filtered_contours_top))

  if len(car_imgs) != 0:
    cv2.imshow('live cars', mosaic_view(car_imgs))
  if len(historic_car_imgs) != 0:
    cv2.imshow('historic', mosaic_view(historic_car_imgs))

  contour_separated = list(map(lambda x: (x, True), filtered_contours_top)) + list(
    map(lambda x: (x, False), filtered_contours_bottom))

  # permite detectar los autos que no tienen contorno asociado
  for car in car_list:
    car.updated = False

  # looping for contours
  for (c, is_top) in contour_separated:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    velocity = 0
    contour_has_matched = False
    for car in car_list:
      distance = calculateDistance(cX, cY, car.x, car.y)
      if distance < 30 and not contour_has_matched and not car.updated:  # es cercano y los considero igual
        contour_has_matched = True
        car.updated = True
        car.x = cX
        car.y = cY
        if (check_velocity): # cada 3 frames
          if (car.hist_x != 0 and car.hist_y != 0):
            velocity = calculate_velocity(cX, cY, car, is_top)
            car.velocity = velocity
            car_img = contour_to_img(clean_frame, c)
            height, width, channels = car_img.shape
            if width > 55 and height < 70:
              car.img = car_img
          car.hist_x = cX
          car.hist_y = cY
        else:
          velocity = car.velocity

    if not contour_has_matched:
      car_list.append(Vehicle(cX, cY, 0))

    draw_contour(c,frame,velocity, is_top, cY)

  for car in car_list:
    if not car.updated:
      car.inactive_counter = car.inactive_counter + 1
    else:
      car.inactive_counter = 0

  old_cars = list(filter(lambda c: c.inactive_counter >= 5, car_list))
  for c in old_cars:
    if c.img.size != 0 and (c.x > 500 or c.x < 150):
      historic_car_imgs.append(c.img)

  car_list = list(filter(lambda c: c.inactive_counter < 5, car_list))

  # cv2.imshow('foreground and background', fgmask)
  # cv2.line(frame, p1, p2, (0, 255, 0), thickness=2)
  check_velocity = False
  cv2.imshow('rgb', frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
