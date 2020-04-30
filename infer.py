import tflite_runtime.interpreter as tflite
import io
from PIL import Image
import numpy as np
from contextlib import contextmanager
import time
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from prometheus_client import Gauge, Summary

import camera

USE_TPU = False
model_file = 'model_quantized_edgetpu.tflite' if USE_TPU else 'model.tflite'
classes = ['clear', 'newpee', 'oldpee', 'poop']
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
EDGETPU_SHARED_LIB = '/usr/lib/libedgetpu.so.1'
REMIND_TIME = 15


INFERENCE_TIME = Summary('inference_time', 'time spent to run inference on a frame')
gauges = {
    klass: Gauge(f'prediction_{klass}', f'prediction score for class {klass}')
    for klass in classes
}

def sprint(*args, **kwargs):
    print(f"[{time.asctime()}]", *args, **kwargs)

@contextmanager
def limit_cpu(pct):
    start = time.time()
    yield
    dur = time.time() - start
    sleep = dur*(1-pct)/pct
    sprint(f"took {dur}s sleeping for {sleep}s")
    time.sleep(sleep)

@contextmanager
def measure(name):
    sprint(f"measuring {name}")
    start = time.time()
    yield
    sprint(f"{name} took {time.time() - start}")


def center_crop(im):
    width, height = im.size   # Get dimensions

    box = min(width, height)

    left = (width - box)/2
    top = (height - box)/2
    right = (width + box)/2
    bottom = (height + box)/2

    # Crop the center of the image
    return im.crop((left, top, right, bottom))


class throttle(object):
    """
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass
    """
    def __init__(self, seconds=0, minutes=0, hours=0):
        self.throttle_period = timedelta(
            seconds=seconds, minutes=minutes, hours=hours
        )
        self.time_of_last_call = datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            time_since_last_call = now - self.time_of_last_call

            if time_since_last_call > self.throttle_period:
                self.time_of_last_call = now
                return fn(*args, **kwargs)

        return wrapper

@throttle(minutes=10)
def handle_newpee():
    os.system('aplay ./WAV/acc_steer_on.wav')


def capture():
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg', use_video_port=True)
    stream.seek(0)
    image = Image.open(stream).convert('RGB')
    size = (224, 224)
    cropped = center_crop(image).resize(size, Image.BILINEAR)
    pix = (((np.array(cropped) / 255) - mean) / std).reshape((1, 224, 224, 3)).astype(np.float32)
    return image, pix


@dataclass
class Capture:
    name: str
    image: Image


def alert_poop():
    os.system('aplay ./WAV/alert_chime.wav')


def handle_poop(recent_images):
    alert_poop()
    for recent in recent_images:
        recent.image.save(recent.name)


def main():
    sprint("loading model...")
    interpreter = tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB, {})] if USE_TPU else None,
    )
    interpreter.allocate_tensors()

    smoothed_class = None
    last_class = None
    class_frame_count = None

    recent_images = deque(maxlen=10)
    last_reminder = 0

    while True:
        image, data = capture()

        with INFERENCE_TIME.time():
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], data)
            interpreter.invoke()
            tflite_results = interpreter.get_tensor(output_details[0]['index'])[0]

        all_classes = list(zip(classes, tflite_results))
        class_idx = np.argmax(tflite_results)
        class_name = classes[class_idx]
        class_prob = tflite_results[class_idx]
        sprint(f"{class_name} ({class_prob}) - {all_classes}")

        for klass, prob in all_classes:
            gauges[klass].set(prob)

        recent_images.append(Capture(name=f"unknown/{camera.now_str()}.jpg", image=image))

        if last_class != class_name:
            last_class = class_name
            class_frame_count = 1
            filename = f"{class_name}.jpg"
            sprint(f"saving {filename}")
            image.save(filename)
        else:
            class_frame_count += 1

        if class_name != smoothed_class and class_frame_count > 5:
            if class_name == 'poop':
                handle_poop(recent_images)
            elif class_name == 'newpee':
                handle_newpee()
            elif smoothed_class == 'poop':
                os.system('aplay ./WAV/acc_steer_off.wav')
            smoothed_class = class_name
            last_reminder = time.time()
            
        if class_name == smoothed_class and class_name == 'poop':
            if last_reminder < (time.time() - REMIND_TIME):
                alert_poop()
                last_reminder = time.time()
