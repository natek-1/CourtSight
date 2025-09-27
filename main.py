from utils.video_utils import read_video, save_video

from tracker.player_tracker import PlayerTracker
from tracker.ball_tracker import BallTracker
from drawers.player_tracks_drawer import PlayerTrackDrawer
from drawers.ball_tracks_drawer import BallTracksDrawer

from ultralytics import YOLO

def main(video_path):
    
    # read video
    video_frames = read_video(video_path)
    
    player_tracker = PlayerTracker("models/yolov8/best.pt", "mps")
    ball_tracker = BallTracker("models/best.pt", "mps")
    
    player_tracks, _ = player_tracker.get_object_track(video_frames,
                                                    stub_path="stubs/player_track_subs.pkl", read_from_stub=True)
    
    
    ball_tracks, _ = ball_tracker.get_object_track(
        video_frames, read_from_stub=True, stub_path="stubs/ball_track_subs.pkl"
    ) 
    
    
    player_drawer = PlayerTrackDrawer()
    ball_drawer = BallTracksDrawer()
    
    # draw track
    video_frames = player_drawer.draw(video_frames, player_tracks)
    video_frames = ball_drawer.draw(video_frames, ball_tracks)
    
    
    
    # save video
    save_video(video_frames, "output_videos/video.mp4")


if __name__ == "__main__":
    main("input_videos/video_1.mp4")


