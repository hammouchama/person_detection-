import cv2 as cv
# ____________________________
p = '../lib/frozen_inference_graph.pb'
v = '../lib/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
det = cv.dnn_DetectionModel(p, v)
# ____________________________
det.setInputSize(320, 230)
det.setInputScale(1.0/127.5)
det.setInputMean((127.5, 127.5, 127.5))
det.setInputSwapRB(True)
# ____________________________
with open('../lib/lable.txt', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# ____________________________

cap = cv.VideoCapture('../video/test2.mp4')
# cap = cv.VideoCapture('../video/vid2.mp4')
# video = cv.VideoWriter('video.avi', fourcc, 25,
#                        (frame.shape[1], frame.shape[0]))
# # ____________________________
# classIds, confs, bbox = det.detect(img, confThreshold=0.5)
# print(classIds, bbox)

while True:
    count = 0
    _, frame = cap.read()
    classIds, confs, bbox = det.detect(frame, confThreshold=0.5)

    for classId, confidens, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if class_names[classId-1] == "person":
            count = count+1
            cv.rectangle(frame, box, color=(255, 0, 0), thickness=2)
            text = " {}".format(count)
            cv.putText(frame, text, (box[0]-10, box[1]-10),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=4)

    c = "Number of person is : {}".format(count)
    cv.putText(frame, c, (10, 30),
               cv.FONT_HERSHEY_DUPLEX, 1, (34, 254, 34), 3)
    cv.imshow("image ", frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
