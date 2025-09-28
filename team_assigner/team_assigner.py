from PIL import Image
import cv2

from transformers import CLIPProcessor, CLIPModel

from utils.stubs_utils import read_stub, save_stub


class TeamAssigner:
    
    def __init__(self,
                 team_1_class="white shirt",
                 team_2_class="blue shirt",
                 device="cpu"):
    
        self.team_colors = {}
        self.player_dict = {}
        self.device = device
        
        self.team_1_class = team_1_class
        self.team_2_class = team_2_class
        self.classes = [team_1_class, team_2_class]
        
        self.processor = None
        self.model = None
    
    def load_model(self):
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(self.device)
    
    def get_player_color(self, frame, bbox):
        
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        image = frame[y1:y2,x1:x2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        inputs = self.processor(text=self.classes, images=image, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)
        
        class_name = self.classes[probs.argmax(dim=1)[0]]
        return class_name

    def get_player_team(self, frame, bbox, player_id):
        
        if player_id in self.player_dict: return self.player_dict[player_id]
        
        player_color = self.get_player_color(frame, bbox)
        team_id = 1 if player_color == self.team_1_class else 2
        
        self.player_dict[player_id] = team_id
        return team_id

    
    def get_player_team_across_frames(self, frames, player_tracks, read_from_stub=False, stub_path=None):
        
        player_assigment = read_stub(read_from_stub=read_from_stub, stub_path=stub_path)
        if player_assigment is not None and len(player_assigment) == len(frames):
            return player_assigment

        self.load_model()
        
        player_assigment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assigment.append({})
            
            if frame_num % 30 == 0: # allow for correction (some caching)
                self.player_dict = {}
            
            
            for player_id, track in player_track.items():
                team = self.get_player_team(
                    frames[frame_num],
                    track["bbox"],
                    player_id=player_id
                )
                player_assigment[frame_num][player_id] = team
                
        if stub_path is not None: save_stub(stub_path, player_assigment)
        
        return player_assigment
        
        
    
    