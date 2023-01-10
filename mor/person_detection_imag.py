import cv2
import numpy as np
import imutils

# load the COCO class names
CLASSES = []
with open('../lib/COCO_labels.txt', 'r') as f:
    class_names = f.read().rstrip('\n').split('\n')

protopath = "../lib/MobileNetSSD_deploy.prototxt"
modelpath = "../lib/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# image = cv2.imread('../images/person_1.jpg')
# image = cv2.imread('../images/person_2.png')
image = cv2.imread('../images/person_3.png')
image = imutils.resize(image, width=600)

(H, W) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 200, True)

detector.setInput(blob)
person_detections = detector.forward()
count = 0
for i in np.arange(0, person_detections.shape[2]):
    confidence = person_detections[0, 0, i, 2]
    if confidence > .2:
        idx = int(person_detections[0, 0, i, 1])
        if class_names[idx] != "person":
            continue
        person_box = person_detections[0, 0,
                                       i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = person_box.astype("int")

        cv2.rectangle(image, (startX, startY),
                      (endX, endY), (0, 0, 255), 2)
        label = " {}".format(count+1)
        cv2.putText(image, label, (startX-2, startY-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        count = count + 1
        # ____________________________________________________
text = "Number of person is : {}".format(count)
cv2.putText(image, text, (10, 30),
            cv2.FONT_HERSHEY_DUPLEX, 1, (34, 254, 34), 2)

cv2.imshow("Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("number of person in this image is ", count)
