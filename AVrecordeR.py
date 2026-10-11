import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import numpy as np
from ultralytics import YOLO

########################
## JRF
## VideoRecorder and AudioRecorder are two classes based on openCV and pyaudio, respectively. 
## By using multithreading these two classes allow to record simultaneously video and audio.
## ffmpeg is used for muxing the two signals
## A timer loop is used to control the frame rate of the video recording. This timer as well as
## the final encoding rate can be adjusted according to camera capabilities
##

########################
## Usage:
## 
## numpy, PyAudio and Wave need to be installed
## install openCV, make sure the file cv2.pyd is located in the same folder as the other libraries
## install ffmpeg and make sure the ffmpeg .exe is in the working directory
##
## 
## start_AVrecording(filename) # function to start the recording
## stop_AVrecording(filename)  # "" ... to stop it
##
##
########################

model = YOLO("yolo26s_int8.onnx", task="detect")

class VideoRecorder():
	
	# Video class based on openCV 
	def __init__(self):
		
		self.open = True
		self.device_index = 0
		self.fps = 0               # fps should be the minimum constant rate at which the camera can
		self.fourcc = "mp4v"       # capture images (with no decrease in speed over time; testing is required)
		self.frameSize = None # video formats and sizes also depend and vary according to the camera used
		self.video_filename = "temp_video.mp4"
		self.video_cap = cv2.VideoCapture(self.device_index)
		self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
		self.frame_counts = 1
		self.framerates = []
		self.frames = []
		self.start_time = time.time()

	# Video starts being recorded 
	def record(self):
     
		while True:
			internal_start_time = time.time()
   
			ret, video_frame = self.video_cap.read()
   
			if ret:
				if self.frameSize is None:
					height, width = video_frame.shape[0], video_frame.shape[1]
					self.frameSize = (width, height)
     
				results = model.track(video_frame, device="cpu", stream=True, verbose=False, classes=[0])
				
				for result in results:
					boxes = result.boxes

					for box in boxes:
						x1, y1, x2, y2 = box.xyxy[0]
						x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
      
						image = cv2.rectangle(video_frame, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)

				internal_end_time = time.time()
				period = internal_end_time - internal_start_time
				fps = int(1 / period)
				self.framerates.append(fps)
				image = cv2.putText(image, f"FPS: {fps}", (1000, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
				# cv2.imshow("frame", image)
				self.frames.append(image)

			else:
				pass

	# Finishes the video recording therefore the thread too
	def stop(self):
		
		if self.open:
			self.fps = int(np.array(self.framerates).mean())
			self.open = False
			video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)

			for frame in self.frames:
				video_out.write(frame)
    
			self.video_out.release()
			self.video_cap.release()
			cv2.destroyAllWindows()
			
		else: 
			pass


	# Launches the video recording function using a thread			
	def start(self):
		video_thread = threading.Thread(target=self.record)
		video_thread.start()


class AudioRecorder():
    
    # Audio class based on pyAudio and Wave
    def __init__(self):
        
        self.open = True
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
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):
        
        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break
        
            
    # Finishes the audio recording therefore the thread too    
    def stop(self):
       
        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
               
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()
        
        pass
    
    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()


def start_AVrecording(filename):
				
	global video_thread
	global audio_thread
	
	video_thread = VideoRecorder()
	audio_thread = AudioRecorder()

	audio_thread.start()
	video_thread.start()

	return filename


def stop_AVrecording(filename):
	
	recorded_fps = video_thread.fps
	video_filename = video_thread.video_filename
	audio_filename = audio_thread.audio_filename
	
	print("Stopping audio...")
	audio_thread.stop() 
	print("Done stopping audio")
 
	print("Stoppping video...")
	video_thread.stop() 
	print("Done stopping video")
 
	mixed_filename = "full_video.mp4"
 
	# Makes sure the threads have finished
	print("Making sure threads have finished...")
	while threading.active_count() > 1:
		time.sleep(1)
	print("Threads have finished")

	
#	 Merging audio and video signal
	
	if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
										
		print ("Re-encoding")
		cmd = f"ffmpeg -r {recorded_fps} -i {video_filename} -pix_fmt yuv420p -r 6 {video_filename}"
		subprocess.call(cmd, shell=True)
	
	else:
		print ("Normal recording")
  
	print ("Mixing")
	cmd = f"ffmpeg -ac 1 -channel_layout stereo -i {audio_filename} -i {video_filename} -pix_fmt yuv420p {mixed_filename}"
	subprocess.call(cmd, shell=True)

	print ("DONE..")




# Required and wanted processing of final files
def file_manager(filename):

	local_path = os.getcwd()

	if os.path.exists(str(local_path) + "/temp_audio.wav"):
		os.remove(str(local_path) + "/temp_audio.wav")
	
	if os.path.exists(str(local_path) + "/temp_video.mp4"):
		os.remove(str(local_path) + "/temp_video.mp4")

	if os.path.exists(str(local_path) + "/full_video.avi"):
		os.remove(str(local_path) + "/full_video.avi")

	if os.path.exists(str(local_path) + "/" + filename + ".avi"):
		os.remove(str(local_path) + "/" + filename + ".avi")
	
	

	
if __name__== "__main__":
	
	filename = "Default_user"	
	file_manager(filename)
	
	start_AVrecording(filename)  
	
	time.sleep(10)
	
	stop_AVrecording(filename)
	print ("Done")
