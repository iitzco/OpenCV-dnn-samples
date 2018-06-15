# Image Classification using GoogLeNet arquitecture trained on ImageNet dataset

import cv2
import sys
import numpy as np

from imagenet_labels import LABEL_MAP

TORCH_MODEL = "./torch/resnet-18.t7"
SIZE = 224

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run(img):
    cvNet = cv2.dnn.readNetFromTorch(TORCH_MODEL)
    img = (img / 255.).astype(np.float32)
    cvNet.setInput(cv2.dnn.blobFromImage(img, 1, (SIZE, SIZE), (0.485, 0.456, 0.406)))
    img = cvNet.forward()
    out = softmax(np.squeeze(img))
    s = np.argsort(out)[-5:]
    for i, each in enumerate(s[::-1]):
        print("{}. {}: {:.2}".format(i+1, LABEL_MAP[each], out[each]))


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    run(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
