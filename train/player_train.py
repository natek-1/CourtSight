import os 
import shutil

from roboflow import Roboflow
from dotenv import load_dotenv
from ultralytics import YOLO

# Load your previously trained model as the starting point for further training
model = YOLO('yolov8l.pt')

# Start the training with your new data
results = model.train(data='./NBA-Game-Detection-1/data.yaml', epochs=500, imgsz=640, batch=32, plots=True, device="cuda")

'''
load_dotenv()

rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
project = rf.workspace("michaeelnguyen").project("nba-game-detection")
version = project.version(1)
dataset = version.download("yolov8")
                

dataset_path = dataset.location.split('/')[-1]
print(dataset.location)

# required for
for type in {"train", "valid"}:
    shutil.move(f"{dataset_path}/{type}", f"datasets/{dataset_path}/{type}")
'''

''' # train for player
yolo task=detect mode=train model=yolov5l6u.pt data=./Basketball-Players-25/data.yaml epochs=100 imgsz=640 plots=True batch=32
'''
''' train for ball detection
yolo task=detect mode=train model=yolov5l6u.pt data=./Basketball-Players-25/data.yaml epochs=250 imgsz=640 plots=True batch=32
'''
