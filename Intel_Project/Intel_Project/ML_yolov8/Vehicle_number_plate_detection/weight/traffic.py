import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import urllib.request
import numpy as np

url = 'http://192.168.79.97/cam-hi.jpg'
cv2.namedWindow("Live Cam Testing", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Cam Testing', 1280, 720)
cv2.moveWindow('Live Cam Testing', 0, 0)
picam2 = cv2.VideoCapture(url)


    
if not picam2.isOpened():
    print("Failed to open the IP camera stream")
    exit()

model = YOLO('/home/raspberry/Intel_Project/ML_yolov8/Vehicle_number_plate_detection/weight/license_plate_detector_best_zoom.pt')
my_file = open("/home/raspberry/Intel_Project/ML_yolov8/Vehicle_number_plate_detection/weight/coco2.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
    results = model.predict(im)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        if d < len(class_list):
            c = class_list[d]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(im, f'{c}', (x1, y1), 1, 1)
        else:
            c = 'Unknown'
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(im, f'{c}', (x1, y1), 1, 1)
    
    cv2.imshow("Live Cam Testing", im)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()