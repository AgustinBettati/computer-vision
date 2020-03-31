import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('img1', frame)
    if cv2.waitKey(1) == ord('h'):
        ticks = str(cv2.getTickCount())
        cv2.imwrite(ticks + '.png', frame)
    if cv2.waitKey(1) == ord('z'):
        break

cap.release()
cv2.destroyAllWindows()
