from ultralytics import YOLO

model = YOLO("helmet.pt")   # path ke file modelmu
model.export(format="onnx", imgsz=640)
