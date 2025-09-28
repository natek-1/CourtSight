from ultralytics import YOLO

# Load your previously trained model as the starting point for further training
model = YOLO('yolov8l.pt')

# Start the training with your new data
results = model.train(data='./NBA-ball-detection-5/data.yaml', epochs=200, imgsz=640, batch=32, plots=True, device="cuda")