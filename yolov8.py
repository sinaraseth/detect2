from ultralytics import YOLO

model = YOLO ('yoloDetect.pt')
model.predict(source=0, imgsz=640, conf=0.6, show=True)