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

'''
for type in {"train", "valid", "test"}:
    shutil.move(f"{dataset_path}/{type}", f"{dataset_path}/{dataset_path}/{type}")
'''

'''
yolo task=detect mode=train model=yolov516u.pt dataset={dataset.location}/data.yaml epochs=100 imsz=640 plots=True batch=16
'''