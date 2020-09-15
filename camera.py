import datetime
import picamera
import threading
from prometheus_client import Summary

camera = picamera.PiCamera()
lock = threading.Lock()

CAPTURE_TIME = Summary("capture_time", "time spent to capture a frame")


@CAPTURE_TIME.time()
def capture(*args, **kwargs):
    lock.acquire()
    ret = camera.capture(*args, **kwargs)
    lock.release()
    return ret


def now_str() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()
