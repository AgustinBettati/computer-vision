import cv2


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
