import os
import cv2
import pytesseract
from ultralytics import YOLO

model = YOLO(r"best_openvino_model\best_openvino_model", task="detect")

def image_loc(foldername):
    fileName = []
    for root, dirs, files in os.walk(os.path.abspath(foldername)):
        for namef in files:
            file_name = os.path.abspath(os.path.join(root, namef))
            fileName.append(file_name)
    return fileName


images = image_loc("images")

def extract_text_from_image(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 8')  
    return text.strip()

for i in images:
    img = cv2.imread(i)

    results = model.predict(img, imgsz=320, half=True, conf=0.5, iou=0.6)
    
    for result in results:
        boxes = result[:5].boxes.numpy() 
        for box in boxes:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            roi = img[y1:y2, x1:x2]
            number_plate_text = extract_text_from_image(roi)
            cv2.putText(img, number_plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.namedWindow("inference", cv2.WINDOW_NORMAL)  
    cv2.imshow('inference', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
