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


def save_video(frames, video_path, fps=30):
    if not os.path.exists(os.path.dirname(video_path)):
        os.mkdir(os.path.dirname(video_path))
    
    height, width, channels = frames[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
        
    out.release()    