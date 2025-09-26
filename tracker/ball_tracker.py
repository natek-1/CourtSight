
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
                         , read_from_stub=False):
    
        tracks = read_stub(stub_path, read_from_stub)

        if tracks is not None and len(tracks) == len(frames): return tracks
        
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

        return tracks