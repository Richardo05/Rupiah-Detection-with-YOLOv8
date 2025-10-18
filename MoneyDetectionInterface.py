from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2


model = YOLO("D:/***/***/***/MoneyDetectionV2.pt") #change path to yout file .pt    
model.predict(source = "1", show = True, conf=0.5)
