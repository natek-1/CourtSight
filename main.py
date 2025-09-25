from ultralytics import YOLO

model = YOLO("yolov8x").to("mps")

result = model.predict("input_videos/video_1.mp4", save=True)
print(result)

for bbox in result[0].boxes:
    print(bbox.xyxy)
