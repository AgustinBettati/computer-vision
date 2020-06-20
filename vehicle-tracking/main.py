import cv2

cap = cv2.VideoCapture('carsRt9_3.avi')
mog2 = cv2.createBackgroundSubtractorMOG2()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while(1):
    ret, frame = cap.read()

    mog2mask = mog2.apply(frame)
    mog2mask = cv2.morphologyEx(mog2mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('mog2 mask', mog2mask)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

