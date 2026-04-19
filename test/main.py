import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("opened:", cap.isOpened())

for i in range(60):
    ret, frame = cap.read()
    if ret and frame is not None:
        print(i, frame.shape, frame.mean())
        cv2.imshow("cam", frame)
        cv2.waitKey(1)
    else:
        print(i, "read failed")
    time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()