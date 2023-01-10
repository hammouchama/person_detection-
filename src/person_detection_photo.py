import cv2 as cv
p = '../lib/frozen_inference_graph.pb'
v = '../lib/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
det = cv.dnn_DetectionModel(p, v)
det.setInputSize(320, 230)
det.setInputScale(1.0/127.5)
det.setInputMean((127.5, 127.5, 127.5))
det.setInputSwapRB(True)
with open('../lib/lable.txt', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
img = cv.imread("../images/person_1.jpg")
# img = cv.imread("../images/person_2.png")
# img = cv.imread("../images/person_2.png")
classIds, confs, bbox = det.detect(img, confThreshold=0.5)
# print(classIds, bbox)
count = 0
for classId, confidens, box in zip(classIds.flatten(), confs.flatten(), bbox):
    if class_names[classId-1] == "person":
        count = count+1
        # cv.rectangle(img, box, color=(255, 0, 0), thickness=2)
        # text = " {}".format(count)
        # cv.putText(img, text, (box[0]-10, box[1]-10),
        #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
text = "Number of person is : {}".format(count)
cv.putText(img, text, (10, 30),
           cv.FONT_HERSHEY_DUPLEX, 1, (34, 254, 34), 2)
cv.imshow("image ", img)
cv.waitKey(0)
