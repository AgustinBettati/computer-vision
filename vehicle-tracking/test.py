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

frame_counter = 0
check_velocity = False

def obtain_color(area):
  if area < 2200:
    return (0, 255, 0)
  if area < 3000:
    return (0,255,255)
  else:
    return (0, 0, 255)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
  w_min = 200
  im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                    for im in im_list]
  return cv2.vconcat(im_list_resize)


def contour_to_img(img, contour):
  (x, y, w, h) = cv2.boundingRect(contour)
  crop_image = img[y:y+h, x:x+w]
  return crop_image


def calculateDistance(x1, y1, x2, y2):
  dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  return dist

while (cap.isOpened):

  import time
  time.sleep(0.100)
  # if ret is true than no error with cap.isOpened
  ret, frame = cap.read()

  frame_counter = frame_counter + 1
  if (frame_counter > 30):  # cada 1s
    check_velocity = True
    frame_counter = 0


  # apply background substraction
  fgmask = fgbg.apply(frame)

  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) # Dilation followed by Erosion
  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) # Erosion followed by dilation

  # check opencv version

  (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # final = vconcat_resize_min(imgs)

  filtered_contours = list(filter(lambda c: cv2.contourArea(c) > 500, contours))
  imgs = list(map(lambda c: contour_to_img(frame, c), filtered_contours))

  # looping for contours
  for c in filtered_contours:

    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    velocity = 0
    has_matched = False

    for car in car_list:
      distance = calculateDistance(cX, cY, car.x, car.y)
      if distance < 30 and not has_matched: #es cercano y los considero igual
        has_matched = True
        car.updated = True
        if(check_velocity):
          if(car.hist_x != 0 and car.hist_y != 0):
            distance = calculateDistance(cX, cY, car.hist_x, car.hist_y)
            distanceM = distance * 0.07
            meterPerSecond = distanceM / 1
            velocity = round(meterPerSecond * 3.6, 2)
            car.velocity = velocity
          car.hist_x = cX
          car.hist_y = cY
        else:
          velocity = car.velocity
        car.x = cX
        car.y = cY

    if not has_matched:
      car_list.append(Vehicle(cX, cY, 0))

    # get bounding box from countour
    (x, y, w, h) = cv2.boundingRect(c)

    # draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), obtain_color(cv2.contourArea(c)), 2)

    cv2.putText(frame, str(velocity), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

  for car in car_list:
    if not car.updated:
      car.inactive_counter = car.inactive_counter + 1

  car_list = list(filter( lambda c: c.inactive_counter < 60, car_list))

  # cv2.imshow('foreground and background', fgmask)
  check_velocity = False
  cv2.imshow('mosaic', vconcat_resize_min(imgs))
  cv2.imshow('rgb', frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
