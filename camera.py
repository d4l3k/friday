import datetime
import cv2
import threading
from PIL import Image
from prometheus_client import Summary

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

lock = threading.Lock()

CAPTURE_TIME = Summary("capture_time", "time spent to capture a frame")


@CAPTURE_TIME.time()
def capture():
    lock.acquire()
    ret, img = cap.read()
    lock.release()
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return image


def now_str() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()

