import capture
import infer
import threading
import os
import traceback
from prometheus_client import start_http_server

threads = []


def thread_wrapper(f):
    try:
        f()
    except Exception:
        traceback.print_exc()
        os._exit(1)


def go(f):
    t = threading.Thread(target=thread_wrapper, args=(f,))
    t.start()
    threads.append(t)

def prometheus():
    start_http_server(2112)

go(capture.main)
go(infer.main)
go(prometheus)

for thread in threads:
    thread.join()
