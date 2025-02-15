from roboflow import Roboflow
import supervision as sv

rf = Roboflow(api_key="riMiboxeLKguBWpBlc7J")
project = rf.workspace().project("brain-tumor-detection-vbopi")
model = project.version(1).model

result = model.predict("b.jpg", confidence=40, overlap=30).json()

detections = sv.Detections.from_roboflow(result)

print(len(detections))

# filter by class
detections = detections[detections.class_id == 0]
print(len(detections) 
