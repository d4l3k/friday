import os

import camera


from evdev import InputDevice, categorize, ecodes, KeyEvent

code_to_category = {
    "KEY_E": "clear",
    "KEY_R": "oldpee",
    "KEY_T": "newpee",
    "KEY_Y": "poop",
}


def main():
    device = InputDevice("/dev/input/event0")  # my keyboard
    print("reading...")
    for event in device.read_loop():
        if event.type == ecodes.EV_KEY:
            desc = categorize(event)
            event = KeyEvent(event)
            time = camera.now_str()
            category = code_to_category.get(event.keycode)
            print("key_pressed:", desc, category)
            if category and event.keystate == KeyEvent.key_down:
                camera.capture().save(f"data/{category}/{time}.jpg")
                os.system("aplay ./WAV/lock_chime.wav")
