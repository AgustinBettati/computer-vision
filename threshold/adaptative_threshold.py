import cv2 as cv

size = 2

size_name = 'Low H'
window_detection_name = 'Object Detection'


def on_size_thresh_trackbar(val):
  global size
  size = val
  cv.setTrackbarPos(size_name, window_detection_name, val)


cap = cv.VideoCapture(0)
cv.namedWindow(window_detection_name)
cv.createTrackbar(size_name, window_detection_name, 0, 1000, on_size_thresh_trackbar)

while True:

    ret, frame = cap.read()
    if frame is None:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    adapt = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, size, 2)
    cv.imshow(window_detection_name, adapt)

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
