# Reference : https://www.youtube.com/watch?v=1LCb1PVqzeY

import cv2
import numpy as np
import time
from show_image import img_show
np.random.seed(42)

N_IMAGES = 5
CONFIDENCE = 0.5
NMS_THRESHOLD = 0.4
class_names_file_path = "data/obj.names"
test_images_file_path = "data/valid.txt"

net = cv2.dnn.readNet("backup/yolov3-tiny-prn_final.weights", "cfg/yolov3-tiny-prn.cfg")

classes = []
with open(class_names_file_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
cap = cv2.VideoCapture('yoko_wakare.mp4')

########## This part below is responsible for the downlod of the model output

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')# note the lower case

width = int(cap.get(3))
height = int(cap.get(4))

out_video = cv2.VideoWriter('video_output.mp4', fourcc , 10, (width,height), True)

##########

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

with open(test_images_file_path, "r") as f:
    paths = np.array([line.strip() for line in f.readlines()])
    images_list = np.random.choice(paths, size=N_IMAGES)

while True:

    # Initalize lists to store detections
    class_ids = []
    confidences = []
    boxes = []

    _,  img = cap.read()
    
    blob = cv2.dnn.blobFromImage(
        img,
        1 / 255.0, (416, 416), (0,0,0),
        swapRB=True, 
        crop=False
    )

    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()
    
    print(f"[INFO] YOLO prediction time: {end - start}")
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                print("[INFO] Object detected!")
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESHOLD)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(indexes) > 0:
        for i in indexes.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[class_ids[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), font, 0.5, (255,255,255), 2)
    
    out_video.write(img)
    cv2.imshow('Image', img)
    
    #key = cv2.waitKey(1)
    if cv2.waitKey(33) == 13:
        break

cap.release()
cv2.destroyAllWindows()