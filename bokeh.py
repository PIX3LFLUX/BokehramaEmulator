# **********************************************************************************************************************
# Jeff Manderscheid, Matriculation Number: 60037
# Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
# Bokehrama Emulation V1.0, 22.09.2020
# **********************************************************************************************************************

import cv2 as cv
import numpy as np


def create_bokeh(image, filtersize=21):
    im = image.copy()
    im = cv.GaussianBlur(im, (filtersize, filtersize), (filtersize - 1.0 / 3.0))
    return im


def create_bokeh_disks(image):
    x, y = image.shape[:2]
    # print('shape: ' + str(x) + ', ' + str(y))
    bokeh = np.zeros(image.shape, np.uint8)
    centres = np.zeros(image.shape, np.uint8)
    for ix in range(int(x/10), x+1, int(x/10)):
        for iy in range(int(y/10), y+1, int(y/10)):
            # print('area max: ' + str(ix) + ', ' + str(iy))
            area = image[ix-int(x/10):ix, iy-int(y/10):iy]
            area_centres = find_centres(area)
            centres[ix-int(x/10):ix, iy-int(y/10):iy] = area_centres
    bokeh = create_disks(centres, image)
    return bokeh


def find_centres(area):
    area_centres = np.zeros(area.shape, np.uint8)
    area_gray = cv.cvtColor(area, cv.COLOR_RGB2GRAY)
    area_gray = cv.GaussianBlur(area_gray, (5, 5), cv.BORDER_DEFAULT)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    max_value = np.max(np.max(area_gray))
    dis = int(np.max(area_gray.shape)/50)
    if np.mean(area_gray) < max_value-40:
        dc, area_gray = cv.threshold(area_gray, int(max_value-10), max_value, cv.THRESH_BINARY)
        area_gray = cv.dilate(area_gray, kernel, iterations=5)
        contours, hierarchy = cv.findContours(area_gray, 1, 2)
        for c in contours:
            mass = cv.moments(c)
            if mass["m00"] != 0.:
                x = int(mass["m10"] / mass["m00"])
                y = int(mass["m01"] / mass["m00"])
                red = np.mean(area[y - dis:y + dis, x - dis:x + dis, 0])
                green = np.mean(area[y - dis:y + dis, x - dis:x + dis, 1])
                blue = np.mean(area[y - dis:y + dis, x - dis:x + dis, 2])
                area_centres[y, x] = (red, green, blue)
    return area_centres


def create_disks(centres, image):
    k_size = int(np.max(image.shape) / 15)
    k2_size = int(np.max(image.shape) / 50)
    g_size = 31
    hb = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    hs = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k2_size, k2_size)) / (k2_size * k2_size)
    blurred = cv.filter2D(image, -1, hs)
    disks = cv. filter2D(centres, -1, hb)
    disks = cv.GaussianBlur(disks, (g_size, g_size), (g_size - 1.0 / 3.0))
    res = cv.addWeighted(blurred, 1, disks, 0.2, 0)
    return res
