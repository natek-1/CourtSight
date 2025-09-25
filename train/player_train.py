import os 
import shutil

from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
project = rf.workspace("roboflow-universe-projects").project("basketball-players-fy4c2")
version = project.version(25)
dataset = version.download("yolov5")

dataset_path = dataset.location.split('/')[-1]
print(dataset.location)

# required for yolov 5
for type in {"train", "valid"}:
    shutil.move(f"{dataset_path}/{type}", f"datasets/{dataset_path}/{dataset_path}/{type}")


''' # train for player
yolo task=detect mode=train model=yolov5l6u.pt data=./Basketball-Players-25/data.yaml epochs=100 imgsz=640 plots=True batch=16
'''
''' train for ball detection
yolo task=detect mode=train model=yolov5l6u.pt data=./Basketball-Players-25/data.yaml epochs=250 imgsz=640 plots=True batch=16
'''