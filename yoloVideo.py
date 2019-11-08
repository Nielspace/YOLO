import numpy as np 
import cv2
import time

net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# print(len(classes))

layer_names = net.getLayerNames()

outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255, size=(len(classes), 3))


cap = cv2.VideoCapture(0)

time_now = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id+=1
    height, width, channels = frame.shape
    frame = cv2.resize(frame, (1000, 1000))

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    class_ids = []
    confidences = []
    boxes = []
    for i in outs:
        for detection in i:
            print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence >0.2:
                #object detected
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)

                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img, (center_x, center_y), 10, (0,0,255), 2)

                #Rectangle coordinates
                x = int(center_x- w/2)
                y = int(center_y- h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # cv2  .rectangle(img, (x,y), (x+w, y+h), (0,255,0))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[i]
            confidence = confidences[i]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y), font, 1, (0,0,0), 2, cv2.LINE_AA)


    elasped = time.time() - time_now
    fps = frame_id/elasped
    cv2.putText(frame, "FPS: " + str(fps), (10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),3 )
    cv2.imshow('Image', frame)
    key=cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.DestroyAllWindows()