import cv2
from tello_zune import TelloZune
import detect_yolo as dy
import tracking
tello = TelloZune(DEBUG=True)

tello.start_tello()

while True:
    tello.start_video()
    img = tello.get_frame()
    dy.start_detection(img)
    tracking.start_tracking(tello, dy.values_detect)
    tello.calc_fps(dy.values_detect[0])
    cv2.imshow("Tello", dy.values_detect[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.end_tello()
cv2.destroyAllWindows()