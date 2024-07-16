# Get full path of HW1
import os
current_file_path = os.path.abspath(__file__)
def joined_path(path):
    return os.path.join(os.path.dirname(current_file_path), path)


class train():
    dataset_path = joined_path("video_datasets/dataset.yaml")  
    train_images_path = joined_path("video_datasets/images/train")
    train_labels_path = joined_path("video_datasets/labels/train")
    val_images_path = joined_path("video_datasets/images/val")
    val_labels_path = joined_path("video_datasets/labels/val")

class cml():
    project_name = "HW1"
    task_name = "less_epochs"

class yolo():
    original_model_path = joined_path("yolov8l.pt")
    image_trained_model_path = joined_path("runs/detect/train14/weights/best.pt")
    video_trained_model_path = joined_path("runs/detect/train46/weights/best.pt")

class inference():
    video_path = joined_path("ood_video_data/surg_1.mp4")
    image_path = joined_path("image_datasets/images/val/de6b6a6c-frame_2683.jpg")
    output_video_path = joined_path("ood_video.mp4")
    output_image_path = joined_path("output.jpg")
    max_frame_count = -1

class config():
    train = train()
    cml = cml()
    yolo = yolo()
    inference = inference()

class confidence():
    threshold = [0.5,0.7,0.6]