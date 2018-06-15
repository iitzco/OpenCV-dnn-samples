# Image Classification using GoogLeNet arquitecture trained on ImageNet dataset

import numpy as np
import sys
import cv2

PROTOTXT = "./caffe/bvlc_googlenet.prototxt"
MODEL = "./caffe/bvlc_googlenet.caffemodel"

from imagenet_labels import LABEL_MAP
SIZE = 224


def run_caffe(net, image, input_size):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (input_size, input_size)), 1,
            (input_size, input_size), (104, 177, 123))

    net.setInput(blob)
    out = net.forward()
    return out



def run(img):
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    preds = run_caffe(net, img, SIZE)

    idxs = np.argsort(preds[0])[::-1][:5]

    for i, idx in enumerate(idxs):
        print("{}. {}: {:.2}".format(i + 1, LABEL_MAP[idx], preds[0][idx]))


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    run(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
