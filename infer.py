import torch
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
from camera import capture, now_str
import cv2

from model import Net, val_transform

torch.backends.quantized.engine = 'qnnpack'

model_file = "./friday_net.pth"
classes = ["clear", "newpee", "oldpee", "poop"]
REMIND_TIME = 15


TRANSFORM_TIME = Summary("transform_time", "time spent to transform a frame")

INFERENCE_TIME = Summary("inference_time", "time spent to run inference on a frame")
gauges = {
    klass: Gauge(f"prediction_{klass}", f"prediction score for class {klass}")
    for klass in classes
}


def sprint(*args, **kwargs):
    print(f"[{time.asctime()}]", *args, **kwargs)


@contextmanager
def limit_cpu(pct):
    start = time.time()
    yield
    dur = time.time() - start
    sleep = dur * (1 - pct) / pct
    sprint(f"took {dur}s sleeping for {sleep}s")
    time.sleep(sleep)


@contextmanager
def measure(name):
    sprint(f"measuring {name}")
    start = time.time()
    yield
    sprint(f"{name} took {time.time() - start}")

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
        self.throttle_period = timedelta(seconds=seconds, minutes=minutes, hours=hours)
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
    os.system("aplay ./WAV/acc_steer_on.wav")


@dataclass
class Capture:
    name: str
    image: Image


def alert_poop():
    os.system("aplay ./WAV/alert_chime.wav")

def boot():
    os.system("aplay ./WAV/start_recording.wav")


def handle_poop(recent_images):
    alert_poop()
    for recent in recent_images:
        recent.image.save(recent.name)


@torch.inference_mode()
def main():
    boot()
    sprint("loading model...")

    if False:
        net = Net()
        net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        net.eval()
    else:
        net = torch.jit.load('friday_net_quant_jit.pt')
        net.eval()

    smoothed_class = None
    last_class = None
    class_frame_count = None

    recent_images = deque(maxlen=10)
    last_reminder = 0

    sprint("starting inference, may take a minute for jit to compile")

    while True:
        image = capture()
        with TRANSFORM_TIME.time():
            data = val_transform(image)

        with INFERENCE_TIME.time():
            out = net(data.unsqueeze(0))

        all_classes = list(zip(classes, out[0]))
        class_idx = out.argmax(1).item()
        class_name = classes[class_idx]
        class_prob = out[0, class_idx]
        sprint(f"{class_name} ({class_prob}) - {all_classes}")

        for klass, prob in all_classes:
            gauges[klass].set(prob)

        recent_images.append(
            Capture(name=f"unknown/{now_str()}.jpg", image=image)
        )

        if last_class != class_name:
            last_class = class_name
            class_frame_count = 1
            filename = f"{class_name}.jpg"
            sprint(f"saving {filename}")
            image.save(filename)
        else:
            class_frame_count += 1

        if class_name != smoothed_class and class_frame_count > 5:
            if class_name == "poop":
                handle_poop(recent_images)
            elif class_name == "newpee":
                handle_newpee()
            elif smoothed_class == "poop":
                os.system("aplay ./WAV/acc_steer_off.wav")
            smoothed_class = class_name
            last_reminder = time.time()

        if class_name == smoothed_class and class_name == "poop":
            if last_reminder < (time.time() - REMIND_TIME):
                alert_poop()
                last_reminder = time.time()

if __name__ == "__main__":
    main()
