import os 
import shutil

from dotenv import load_dotenv

from roboflow import Roboflow



load_dotenv()

rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
project = rf.workspace("cps-etiji").project("nba-ball-detection")
version = project.version(5)
dataset = version.download("yolov8")

dataset_path = dataset.location.split('/')[-1]
print(dataset.location)

# required for
for type in {"train", "valid"}:
    shutil.move(f"{dataset_path}/{type}", f"datasets/{dataset_path}/{type}")

                