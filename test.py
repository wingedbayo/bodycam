import os
import subprocess
import threading
import time
import wave

import cv2
import numpy as np
import pyaudio
from ultralytics import YOLO

model = YOLO("yolo26s_int8.onnx", task="detect")


class VideoRecorder:
    def __init__(self, device_index=0):
        self.video_filename = "temp_video.mp4"
        self.fps = 0
        self.frame_size = None
        self.frames = []
        self.timestamps = []
        self.stop_event = threading.Event()
        self.thread = None
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    def record(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if self.frame_size is None:
                h, w = frame.shape[:2]
                self.frame_size = (w, h)

            t0 = time.time()
            results = model.predict(frame, device="cpu", verbose=False, classes=[0])
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            fps = 1.0 / max(time.time() - t0, 1e-6)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

            self.frames.append(frame)
            self.timestamps.append(time.time())

    def start(self):
        self.thread = threading.Thread(target=self.record, daemon=True)
        self.thread.start()

    def stop(self):
        # 1. stop the thread and WAIT for it before releasing anything
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.cap.release()

        if len(self.frames) < 2:
            print("No frames captured")
            return

        # 2. real fps = frames / elapsed time
        duration = self.timestamps[-1] - self.timestamps[0]
        self.fps = max(len(self.frames) / duration, 1.0)

        # 3. write the file once, with valid fps and size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.video_filename, fourcc, self.fps, self.frame_size)
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        self.frames.clear()


class AudioRecorder:
    def __init__(self):
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.frames_per_buffer)
        self.audio_frames = []
        self.stop_event = threading.Event()
        self.thread = None

    def record(self):
        while not self.stop_event.is_set():
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            self.audio_frames.append(data)

    def start(self):
        self.thread = threading.Thread(target=self.record, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()

        with wave.open(self.audio_filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.audio_frames))
        self.audio.terminate()


video_rec = None
audio_rec = None


def start_AVrecording(filename):
    global video_rec, audio_rec
    video_rec = VideoRecorder()
    audio_rec = AudioRecorder()
    audio_rec.start()
    video_rec.start()
    return filename


def stop_AVrecording(filename):
    # Use the SAME instances that were started (don't create new ones)
    audio_rec.stop()
    video_rec.stop()

    mixed_filename = f"{filename}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_rec.video_filename,
        "-i", audio_rec.audio_filename,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        mixed_filename,
    ]
    print("Mixing")
    subprocess.call(cmd)


def file_manager(filename):
    for f in ("temp_audio.wav", "temp_video.mp4", f"{filename}.mp4"):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    filename = "Default_user"
    file_manager(filename)

    start_AVrecording(filename)
    time.sleep(10)
    stop_AVrecording(filename)
    print("Done")