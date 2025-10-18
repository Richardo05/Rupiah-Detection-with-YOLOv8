from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2


model = YOLO("D:/uni/sem 3/AI/MoneyDetectionV2.pt") #ganti jadi path ke file .pt    
model.predict(source = "1", show = True, conf=0.5)

# D:/uni/sem 3/AI/video-rupiah.mp4