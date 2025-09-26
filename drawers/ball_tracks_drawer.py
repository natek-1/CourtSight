from drawers.utils import draw_triangle


class BallTracksDrawer:
    
    def __init__(self, color=(0, 255, 0)):
        self.color = color

    
    def draw(self, frames, tracks):
        
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            frame=frame.copy()
            track_info= tracks[frame_num]
            
            for ball in track_info.values():
                frame = draw_triangle(frame, ball['bbox'], self.color)
            
            output_frames.append(frame)
        
        return output_frames
                
        