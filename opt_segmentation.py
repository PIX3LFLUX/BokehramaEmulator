# **********************************************************************************************************************
# Jeff Manderscheid, Matriculation Number: 60037
# Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
# Bokehrama Emulation V1.0, 22.09.2020
# **********************************************************************************************************************

import cv2 as cv
import numpy as np
import random


def canny(image):
    im = image.copy()
    med = np.median(im)
    sig = 0.33
    low = int(max(0, (1.0 - sig) * med))

    upp = int(max(255, (1.0 + sig) * med))
    can = cv.Canny(im, low, upp)
    kernel = np.ones((3, 3))
    can = cv.dilate(can, kernel)
    return can


def opening(image, iteration=10, kernel_size=10):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in range(iteration):
        image = cv.dilate(image, kernel)
        image = cv.erode(image, kernel)
    return image


def closing(image, iteration=10, kernel_size=10):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in range(iteration):
        image = cv.erode(image, kernel)
        image = cv.dilate(image, kernel)
    return image


def delete_holes(image):
    h, w = image.shape[:2]
    a = np.zeros(image.shape, np.dtype('uint8'))
    b = np.zeros(image.shape, np.dtype('uint8'))
    can = canny(image)
    a = can.copy()
    masking = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(image=can, mask=masking, seedPoint=(0, 0), newVal=255)
    b = cv.bitwise_xor(can, a)
    b = cv.bitwise_not(b)
    return b


def filler(edge, mask, no_zero, iterations=100):
    result = np.zeros(mask.shape, np.dtype('uint8'))
    h, w = mask.shape[:2]
    for i in range(iterations):
        temp = edge.copy()
        rand = random.randrange(len(no_zero[0]))
        h_rand = no_zero[0][rand]
        w_rand = no_zero[1][rand]

        masking = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(image=edge, mask=masking, seedPoint=(w_rand, h_rand), newVal=255)
        diff = cv.bitwise_xor(edge, temp)
        result = cv.add(result, diff)

        mask = cv.subtract(mask, diff)
        no_zero = np.nonzero(mask)
    return result


def enhance_seg(image, mask, open=False, del_holes=False):
    # running edge detection
    edge = canny(image)
    # creating a smaller and a bigger mask
    mask = closing(mask)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    mask_small = cv.erode(mask, kernel, iterations=5)
    mask_big = cv.dilate(mask, kernel, iterations=2)
    # eliminate the edges as possible seeds
    mask_small = cv.subtract(mask_small, edge)
    # running the filler algorithm
    filled = filler(edge, mask_small, np.nonzero(mask_small))
    # big mask defines the maximum where the object could be
    # when edges arent closed, floodfill can fill entire areas not belonging to the object
    filled = cv.bitwise_and(filled, mask_big)
    if open:
        filled = opening(filled)
    if del_holes:
        filled = delete_holes(filled)
    return filled
