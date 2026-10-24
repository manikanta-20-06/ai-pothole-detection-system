from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Point directly to the data.yaml inside your archive folder
model.train(
    data=r"archive\data.yaml",
    epochs=25,
    imgsz=640
)