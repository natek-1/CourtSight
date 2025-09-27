import numpy as np

import pandas as pd

from ultralytics import YOLO
import supervision as sv

from utils.stubs_utils import read_stub, save_stub


class BallTracker:
    
    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model_path).to(device)
        
    def detect_frames(self, frames, batch_size=32, conf=0.5):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            predictions = self.model.predict(batch_frames, conf=conf)
            detections += predictions
        return detections

    def get_object_track(self, frames, batch_size=32, conf=0.5, stub_path=None
                         , read_from_stub=False, detections=None):
    
        tracks = read_stub(stub_path, read_from_stub)

        if tracks is not None and len(tracks) == len(frames): return tracks, None
        
        if detections is None:
            detections = self.detect_frames(frames, batch_size=batch_size, conf=conf)
            
        tracks = []
        
        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_to_idx = {classification:idx for idx,classification in cls_name.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({}) # base case if not ball detected, empty dict
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_idx = frame_detection[3]
                confidence = frame_detection[2]
                
                if cls_idx == cls_to_idx['Ball']: # in case multiple ball detected select the one with highest confidence
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence
            
            if chosen_bbox is not None: # id of ball is 1
                tracks[frame_num][1] = {'bbox': chosen_bbox}
        
        if stub_path is not None: save_stub(stub_path, tracks)

        return tracks, detections

    def remove_wrong_detections(self, ball_postions):
        
        MAXIMUM_ALLOWED_DISTANCE = 30
        last_good_idx = -1
        
        for idx, position in enumerate(ball_postions):
            current_box = position.get(1, {}).get('bbox', [])
            
            if len(current_box) == 0: continue # no detection occured
            if last_good_idx == -1:
                last_good_idx = idx # first valid detection
                continue
            
            last_good_box = ball_postions[last_good_idx].get(1, {}).get('bbox', [])
            frame_gap = idx - last_good_idx
            adjusted_max_distance = MAXIMUM_ALLOWED_DISTANCE * frame_gap
            
            if np.linalg.norm(np.array(current_box[:2]) - np.array(last_good_box[:2])) > adjusted_max_distance:
                ball_postions[idx] =  {}
            else:
                last_good_idx = idx
        
        return ball_postions

    
    def interpolate_frame_position(self, ball_positions):
        
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        
        df = df.interpolate()
        df = df.bfill()
        
        ball_positions = [{1: {"bbox": x}} for x in df.to_numpy().tolist()]
        return ball_positions
            
            
        