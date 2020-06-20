import numpy as np
import cv2
import sys

video_path = 'carsRt9_3.avi'
cv2.ocl.setUseOpenCL(False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# read video file
cap = cv2.VideoCapture(video_path)

# check opencv version
fgbg = cv2.createBackgroundSubtractorMOG2()



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


while (cap.isOpened):

  import time
  time.sleep(0.100)
  # if ret is true than no error with cap.isOpened
  ret, frame = cap.read()

  if ret == True:

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
    for c in contours:
      if cv2.contourArea(c) < 500:
        continue

      # get bounding box from countour
      (x, y, w, h) = cv2.boundingRect(c)

      # draw bounding box
      cv2.rectangle(frame, (x, y), (x + w, y + h), obtain_color(cv2.contourArea(c)), 2)

    # cv2.imshow('foreground and background', fgmask)
    cv2.imshow('mosaic', vconcat_resize_min(imgs))
    cv2.imshow('rgb', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()
