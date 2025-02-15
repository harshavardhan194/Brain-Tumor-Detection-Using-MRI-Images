
from roboflow import Roboflow
rf = Roboflow(api_key="riMiboxeLKguBWpBlc7J")
project = rf.workspace().project("brain-tumor-detection-vbopi")
model = project.version(1).model

# infer on a local image
print(model.predict("b.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("b.jpg", confidence=40, overlap=30).save("bc.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())