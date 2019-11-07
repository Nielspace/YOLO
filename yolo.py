import numpy as np 
import cv2

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# print(len(classes))

layer_names = net.getLayerNames()

outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255, size=(len(classes), 3))


# loading image
img = cv2.imread('bike.jpg')
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

#detecting Image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
for b in blob:
    for n,img_blob in enumerate(b):
        cv2.imshow(str(n), img_blob)


net.setInput(blob)
outs = net.forward(outputlayers)

#showing information on the screen
class_ids = []
confidences = []
boxes = []
for i in outs:
    for detection in i:
        print(detection)
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence >0.7:
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

# print(len(boxes))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)
print(indexes)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = classes[class_ids[i]]
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color,2)
        cv2.putText(img, label, (x, y), font, 1, (0,0,0), 2, cv2.LINE_AA)



cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.DestroyAllWindows()


print('executed')