import os

import cv2

def read_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    return frames

def save_video(frames, output_path, fps=30.0):
    os.path.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    
    
    