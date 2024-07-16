# Run predictions on video
from ultralytics import YOLO
from configurations import config

# Get the full video path
full_video = config.inference.video_path

# Load a model
model = YOLO(config.yolo.video_trained_model_path)  

# Run prediction on the frame
results = model.predict(source=full_video, conf=0.1,device=0, save_txt=True,save_conf= True, vid_stride=1,save=True)
