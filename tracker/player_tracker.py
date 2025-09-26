

from ultralytics import YOLO
import supervision as sv

from utils.stubs_utils import read_stub, save_stub

class PlayerTracker:
    
    def __init__(self, model_path, device):
        self.model = YOLO(model_path).to(device)
        self.tracker = sv.ByteTrack()
        
    
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
            
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks.append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if class_id == cls_to_idx['Player']:
                    tracks[frame_num][track_id] = {"bbox":bbox}
        
        if stub_path is not None: save_stub(stub_path, tracks)
        
        return tracks
        