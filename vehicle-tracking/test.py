import math

import numpy as np
import cv2
import sys

from vehicle import Vehicle

video_path = 'carsRt9_3.avi'
cv2.ocl.setUseOpenCL(False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# read video file
cap = cv2.VideoCapture(video_path)

# check opencv version
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=16)

car_list = []
old_car_imgs = []

frame_counter = 0
check_velocity = False

def obtain_color(area, is_top):
  if (is_top and area < 1300) or (not is_top and area < 1500):
    return (0, 255, 0)
  if (is_top and area < 2500) or (not is_top and area < 3000):
    return (0,255,255)
  else:
    return (0, 0, 255)

def mosaic_view(im_list, interpolation=cv2.INTER_CUBIC):
  w_min = 200
  im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                    for im in im_list]
  if len(im_list) > 25:
    first_col = cv2.vconcat(im_list_resize[:7])
    second_col = cv2.vconcat(im_list_resize[7:14])
    third_col = cv2.vconcat(im_list_resize[14:21])
    forth_col = cv2.vconcat(im_list_resize[21:])
    result = [cv2.resize(im, (w_min, 800), interpolation=interpolation)
              for im in [first_col, second_col, third_col, forth_col]]
    return cv2.hconcat(result)
  if len(im_list) > 18:
    first_col = cv2.vconcat(im_list_resize[:7])
    second_col = cv2.vconcat(im_list_resize[7:14])
    third_col = cv2.vconcat(im_list_resize[14:])
    result = [cv2.resize(im, (w_min, 800), interpolation=interpolation)
              for im in [first_col, second_col, third_col]]
    return cv2.hconcat(result)
  elif len(im_list) > 10:
    first_col = cv2.vconcat(im_list_resize[:7])
    second_col = cv2.vconcat(im_list_resize[7:])
    result = [cv2.resize(im, (w_min, 800), interpolation=interpolation)
                      for im in [first_col, second_col]]
    return cv2.hconcat(result)
  else:
    return cv2.vconcat(im_list_resize)


def contour_to_img(img, contour):
  (x, y, w, h) = cv2.boundingRect(contour)
  crop_image = img[y:y+h, x:x+w]
  return crop_image


def calculateDistance(x1, y1, x2, y2):
  dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return dist


def obtain_center(contour):
  M = cv2.moments(contour)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  return [cX, cY]

p1 = (0,190)
p2 = (640,174)
def mutate_y_value(val, flip):
  (y_index, x_array) = val
  for x_val in enumerate(x_array):
    (x_index, value) = x_val
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

while (cap.isOpened):
  import time
  # time.sleep(0.100)
  ret, frame = cap.read()
  clean_frame = frame.copy()

  frame_counter = frame_counter + 1
  if (frame_counter > 3):  # cada 0.1s
    check_velocity = True
    frame_counter = 0


  # apply background substraction
  fgmask = fgbg.apply(frame)

  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) # Dilation followed by Erosion
  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) # Erosion followed by dilation

  top_half = fgmask.copy()
  for val in enumerate(top_half):
    mutate_y_value(val, True)

  bottom_half = fgmask.copy()
  for val in enumerate(bottom_half):
    mutate_y_value(val, False)

  (top_contours, hierarchy) = cv2.findContours(top_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  (bottom_contours, hierarchy) = cv2.findContours(bottom_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  def valid_countor(c):
    (x, y, width, h) = cv2.boundingRect(c)
    return (cv2.contourArea(c) > 500 and width < 150)
  filtered_contours_top = list(filter(lambda c: valid_countor(c), top_contours))
  filtered_contours_bottom = list(filter(lambda c: valid_countor(c), bottom_contours))

  car_imgs = list(map(lambda c: contour_to_img(frame, c), filtered_contours_bottom + filtered_contours_top))

  if (len(car_imgs) != 0):
    cv2.imshow('live cars', mosaic_view(car_imgs))
  if len(old_car_imgs) != 0:
    cv2.imshow('historic', mosaic_view(old_car_imgs))

  for car in car_list:
    car.updated = False

  filtered_contours = list(map(lambda x: (x, True), filtered_contours_top)) + list(map(lambda x: (x, False), filtered_contours_bottom))

  # looping for contours
  for (c, is_top) in filtered_contours:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    velocity = 0
    has_matched = False

    for car in car_list:
      distance = calculateDistance(cX, cY, car.x, car.y)
      if distance < 30 and not has_matched and not car.updated: #es cercano y los considero igual
        has_matched = True
        car.updated = True
        car.x = cX
        car.y = cY
        if(check_velocity):
          if(car.hist_x != 0 and car.hist_y != 0):
            distance = calculateDistance(cX, cY, car.hist_x, car.hist_y)
            global distanceM # en 0.1s
            if is_top:
              distanceM = distance * 0.102
            else:
              distanceM = distance * 0.07
            meterPerSecond = distanceM / 0.1
            velocity = int(meterPerSecond * 3.6)
            car.velocity = velocity
            car_img = contour_to_img(clean_frame, c)
            height, width, channels = car_img.shape
            if(width > 50 and height < 70):
              car.img = car_img
          car.hist_x = cX
          car.hist_y = cY
        else:
          velocity = car.velocity


    if not has_matched:
      car_list.append(Vehicle(cX, cY, 0))

    # get bounding box from countour
    (x, y, w, h) = cv2.boundingRect(c)

    # draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), obtain_color(cv2.contourArea(c), is_top), 2)

    cv2.putText(frame, str(velocity) + " km/h", (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

  for car in car_list:
    if not car.updated:
      car.inactive_counter = car.inactive_counter + 1
    else:
      car.inactive_counter = 0

  old_cars = list(filter( lambda c: c.inactive_counter >= 5, car_list))
  for c in old_cars:
    if c.img.size != 0:
      old_car_imgs.append(c.img)

  car_list = list(filter( lambda c: c.inactive_counter < 5, car_list))

  # cv2.imshow('foreground and background', fgmask)
  # cv2.line(frame, p1, p2, (0, 255, 0), thickness=2)
  check_velocity = False
  cv2.imshow('rgb', frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
