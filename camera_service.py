"""Single-camera background capture and recognition service for Flask."""
import threading
import time


class CameraService:
    def __init__(self):
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._capture = None
        self._jpeg = None
        self._state = {"running": False, "error": None, "latest": None}

    def start(self, camera_index, callback):
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("An attendance camera session is already running.")
            self._stop.clear()
            self._state = {"running": True, "error": None, "latest": None}
            self._thread = threading.Thread(target=self._run, args=(camera_index, callback), daemon=True)
            self._thread.start()

    def _run(self, camera_index, callback):
        try:
            import cv2
            cap = cv2.VideoCapture(camera_index)
            with self._lock:
                self._capture = cap
            if not cap.isOpened():
                raise RuntimeError("Could not open camera. Check its index, connection, and permissions.")
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera stopped providing frames.")
                result = callback(frame)
                if result.get("box"):
                    x, y, w, h = result["box"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 200, 80), 2)
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    with self._lock:
                        self._jpeg = encoded.tobytes()
                        self._state["latest"] = result
                time.sleep(0.04)
        except Exception as exc:
            with self._lock:
                self._state["error"] = str(exc)
        finally:
            with self._lock:
                if self._capture:
                    self._capture.release()
                self._capture = None
                self._state["running"] = False

    def stop(self):
        self._stop.set()
        with self._lock:
            cap = self._capture
        if cap:
            cap.release()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self._lock:
            self._state["running"] = False

    def frame(self):
        with self._lock:
            return self._jpeg

    def state(self):
        with self._lock:
            return dict(self._state)


camera_service = CameraService()
