# Run predictions on video
import cv2
from ultralytics import YOLO
from configurations import config 



# Load a model
model = YOLO(config.yolo.video_trained_model_path)  # load a custom model

# Load image
frame = cv2.imread(config.inference.image_path)

# Run prediction on the frame
results = model.predict(source=frame, conf=0.25,device=0,save_txt=True,save=True, save_conf=True)

