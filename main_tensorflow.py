# Object Detection using SSD Inception arquitecture trained on COCO dataset
import cv2
import sys

FROZEN_GRAPH = "./tensorflow/ssd_inception_v2_coco.pb"
PB_TXT = "./tensorflow/ssd_inception_v2_coco.pbtxt"
SIZE = 300

from coco_labels import LABEL_MAP

def run(img):
    cvNet = cv2.dnn.readNetFromTensorflow(FROZEN_GRAPH, PB_TXT)

    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv2.dnn.blobFromImage(img, 1.0/127.5, (SIZE, SIZE), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            cv2.putText(img, LABEL_MAP[int(detection[1])], (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


    return img


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    image = run(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
