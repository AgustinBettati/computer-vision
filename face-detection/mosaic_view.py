import cv2
import numpy as np


def mosaic_view(img_with_person_list, interpolation=cv2.INTER_CUBIC):
  w_min = 400
  im_list_resize = [(cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation), person)
                    for (im,person) in img_with_person_list]
  resulting_imgs = []
  for (im, person) in im_list_resize:
    h, w = im.shape[:2]
    # create same size image of background color
    bg_color = (0,0,0)
    bg = np.full((im.shape), bg_color, dtype=np.uint8)
    cv2.putText(bg, person.name, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    x, y, w, h = cv2.boundingRect(bg[:, :, 2])
    result = im.copy()
    result[y:y + h, x:x + w] = bg[y:y + h, x:x + w]
    resulting_imgs.append(result)

  return cv2.vconcat(resulting_imgs)

def concat_face_imgs(person, interpolation=cv2.INTER_CUBIC):
  w = 200
  h = 200
  im_list = [person.img, person.video_img]
  im_list_resize = [cv2.resize(im, (w, h), interpolation=interpolation)
                    for im in im_list]
  result = cv2.hconcat(im_list_resize)

  return result
