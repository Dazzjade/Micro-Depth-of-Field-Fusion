#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import EF_utils as EF
import time

K = np.array([[0, -1, 0],
              [-1, 4, -1],
              [0, -1, 0]])


def sharpening1(img):
    B = cv.filter2D(img[:, :, 0], cv.CV_32F, K,
                    borderType=cv.BORDER_REFLECT_101)
    G = cv.filter2D(img[:, :, 1], cv.CV_32F, K,
                    borderType=cv.BORDER_REFLECT_101)
    R = cv.filter2D(img[:, :, 2], cv.CV_32F, K,
                    borderType=cv.BORDER_REFLECT_101)
    B = img[:, :, 0]+B
    G = img[:, :, 1]+G
    R = img[:, :, 2]+R
    return cv.merge([B, G, R])


def sharpening2(img):
    B = cv.GaussianBlur(img[:, :, 0], (3, 3), 0,
                        borderType=cv.BORDER_REFLECT_101)
    G = cv.GaussianBlur(img[:, :, 1], (3, 3), 0,
                        borderType=cv.BORDER_REFLECT_101)
    R = cv.GaussianBlur(img[:, :, 2], (3, 3), 0,
                        borderType=cv.BORDER_REFLECT_101)
    B = img[:, :, 0]-B
    G = img[:, :, 1]-G
    R = img[:, :, 2]-R
    B = img[:, :, 0]+B
    G = img[:, :, 1]+G
    R = img[:, :, 2]+R
    return cv.merge([B, G, R])


def constract_weight(img):
    img = np.float32(img)/255.
    B = np.absolute(cv.Laplacian(img[:, :, 0], cv.CV_32F))
    G = np.absolute(cv.Laplacian(img[:, :, 1], cv.CV_32F))
    R = np.absolute(cv.Laplacian(img[:, :, 2], cv.CV_32F))
    result = cv.merge([B, G, R])
    return result


def exposure_fusion(locations):
    imgs = EF.read_sequence_to_fuse(locations)
    for i in range(len(imgs)):
        imgs[i] = sharpening1(imgs[i])
    a, b, c, d = imgs.shape
    Cweights = np.zeros((a, b, c, d))
    # to fuse pictures with depth of fields' difference only, constract can be the sole measure used
    for i in range(len(imgs)):
        Cweights[i] = constract_weight(imgs[i])
    for i in range(3):
        Cweights[:, :, :, i] = Cweights[:, :, :, i]+1e-12
        Cweights[:, :, :, i] = np.float32(np.einsum(
            'ij,lij->lij', 1./(Cweights[:, :, :, i].sum(axis=0)), Cweights[:, :, :, i]))

    rst = np.zeros(imgs[0].shape, dtype=np.float32)
    for i in range(len(Cweights)):
        imgs[i, :, :] = cv.multiply(
            imgs[i, :, :], Cweights[i], dtype=cv.CV_32FC3)
        rst = rst+imgs[i]
    return rst


now = time.time()
final = exposure_fusion(["align_0000.tif", "align_0001.tif", "align_0002.tif", "align_0003.tif", "align_0004.tif", "align_0005.tif", "align_0006.tif",
                        "align_0007.tif", "align_0008.tif", "align_0009.tif", "align_0010.tif", "align_0011.tif", "align_0012.tif", "align_0013.tif", "align_0014.tif"])/255.
now = time.time()-now
print(now)
# final=np.uint8(exposure_fusion(["0.tif","1.tif","2.tif"]))

# cv.namedWindow("2",cv.WINDOW_NORMAL)
# cv.imshow("2",final)
# cv.waitKey()

# cv.destroyAllWindows()
# cv.imwrite("output_Laplacian_float32_sharpening_0.2.jpg",final*255.)
