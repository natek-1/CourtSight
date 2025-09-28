from drawers.utils import draw_ellipse

class PlayerTrackDrawer:
    
    def __init__(self, team_1_color=[255, 245, 238],team_2_color=[128, 0, 0]):
        self.default_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        pass
    
    def draw(self, video_frames, tracks, player_assignment):
        
        output_video_frames = []
        
        for frame_number, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks[frame_number]
            player_assignment_frame = player_assignment[frame_number]
            
            for track_id, player in player_dict.items():
                team_id = player_assignment_frame[track_id] # since each prediction is from the track, each player(track_id) has a prediction
                color = self.team_1_color if team_id == 1 else self.team_2_color
                frame = draw_ellipse(frame, player['bbox'], color, track_id=track_id)
            output_video_frames.append(frame)
            
        return output_video_frames
            
            