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
import threading


def main():
    USE_TPU = True
    model_file = "model_quantized_edgetpu.tflite" if USE_TPU else "model.tflite"
    classes = ["clear", "newpee", "oldpee", "poop"]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    EDGETPU_SHARED_LIB = "/usr/lib/libedgetpu.so.1"
    REMIND_TIME = 15

    image = Image.open("test.jpg").convert("RGB")
    size = (224, 224)
    cropped = image.resize(size, Image.BILINEAR)
    data = (
        (((np.array(cropped) / 255) - mean) / std)
        .reshape((1, 224, 224, 3))
        .astype(np.float32)
    )

    print("loading model...")
    interpreter = tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB, {})]
        if USE_TPU
        else None,
    )
    interpreter.allocate_tensors()

    print("inferring")

    while True:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], data)
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]["index"])[0]

        all_classes = list(zip(classes, tflite_results))
        class_idx = np.argmax(tflite_results)
        class_name = classes[class_idx]
        class_prob = tflite_results[class_idx]
        print(f"{class_name} ({class_prob}) - {all_classes}")


a = threading.Thread(target=main)
a.start()
a.join()
