import cv2
import os
from datetime import datetime

class ClipRecorder:
    def __init__(self, output_dir="../data/clips", max_frames=80):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_frames = max_frames
        self.frames = []
        self._last_full_msg_time = None

    def add_frame(self, frame):
        from time import time
        now = time()
        if len(self.frames) < self.max_frames:
            self.frames.append(frame)
        else:
            if self._last_full_msg_time is None or (now - self._last_full_msg_time) >= 1.5:
                print("Alcanzado el número máximo de frames de clip")
                self._last_full_msg_time = now

    def save_clip(self):
        if not self.frames:
            print("No hay frames para guardar")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = os.path.join(self.output_dir, f"clip_{timestamp}.avi")
        h, w, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(clip_path, fourcc, 20.0, (w, h))
        for f in self.frames:
            out.write(f)
        out.release()
        self.frames = []
        print(f"Clip guardado en {clip_path}")