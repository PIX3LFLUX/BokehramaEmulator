# **********************************************************************************************************************
# Jeff Manderscheid, Matriculation Number: 60037
# Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
# Bokehrama Emulation V1.0, 22.09.2020
# **********************************************************************************************************************

import cv2
import numpy as np
from fileHandler import save_images
import argparse


# multiple bokehramas: the path to the image folder as an argument
def parse_args():
    parser = argparse.ArgumentParser(description="Create Bokehrama from image, depth map and object mask")
    parser.add_argument('--path', type=str, help='Path of the folder with the images', required=False,
                        default='/home/jeff/PycharmProjects/BokehramaEmulation/images')
    return parser.parse_args()


# edge detection using the canny algorithm
# saves the images into the subfolder /masks_maps/
def detect_edge(images):
    results = []
    for image in images:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        v = np.median(img)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(max(255, (1.0 + sigma) * v))
        result = cv2.Canny(img, lower, upper)
        results.append(result)
    save_images(results, images, "cluster", folder=True)


# Blending function: combines a blurred image with the original using the adjusted depth map
def blender(im1, blur, d_map, exponent=5):
    # converts the adjusted depth map to an alpha array
    # alpha are numbers between 0 and 1
    # close to 1 = sharp, close to 0 = blurred
    alpha = np.exp(-exponent * (1 - (cv2.cvtColor(d_map, cv2.COLOR_GRAY2RGB) / 255.0)))
    blended = cv2.convertScaleAbs(im1*alpha + blur*(1-alpha))
    return blended


# Adjust depth map using the focus depth
def adj_depth(depth, focus_depth):
    # depth map: the higher the number the closer the object
    # shifts the max (255) to the focus depth
    # the number will decrease from the focus depth in both direction, back and front

    if focus_depth < 130:
        # depth doesnt go to 0 for the furthest distance
        depth = np.abs(np.ndarray.astype(depth, dtype='int32') + (255 - focus_depth))
    else:
        # the depth goes to 0 for the furthest distance
        depth = np.floor((np.ndarray.astype(depth, dtype='float32')/focus_depth)*255)

    depth[depth > 255] = 255 - (depth[depth > 255] - 255)
    depth = depth.astype('uint8')
    return depth


# create a mask out of a Photoshop image
def create_mask(im, threshold=255):
    im_out = np.zeros(im.shape, np.dtype('uint8'))
    im_out[im < threshold] = 255
    return im_out


# create mask from segmentation image
def create_seg_mask(seg, square=False):
    x, y = seg.shape
    mask = np.zeros(seg.shape, np.dtype('uint8'))
    if square:
        box = seg[int(x/2 - x/20):int(x/2 + x/20), int(y/2 - y/20):int(y/2 + y/20)]
        pxs = []
        for i in range(box.shape[0]):
            pxs.append(np.bincount(box[i]).argmax())
        px = np.bincount(pxs).argmax()
        mask[seg == px] = 255
    else:
        # simply assumes that the object of interest is in the middle of the image
        px = seg[int(x/2), int(y/2)]
        # creates a binary mask of the object
        mask[seg == px] = 255
    return mask


# determine depth of object in focus
def obj_focus(d_map, mask, avg=False):
    # takes the depth information of the object and averages it to calculate the depth of the object
    # due to the inaccuracy of the depth map when it comes to objects, looking at the average is too inaccurate
    if avg:
        map2 = cv2.bitwise_and(d_map, mask)

        # focus depth from minimum depth of the object
        focus = int(np.mean(map2[np.nonzero(map2)]))

        # focus depth from mean depth of entire object
        # focus = np.mean(map2[map2 > 0])

        # giving the entire object the same depth as the focus/mean depth
        d_map[map2 > 0] = focus
    # the depth information at the bottom of the object is usually quiet accurate
    # therefore only the lowest part of the depth information from object is used
    # the following finds this lowest part in the mask, then averages the depth only in that area
    else:
        map2 = cv2.bitwise_and(d_map, mask)
        [y, x] = map2.shape
        area = [0, 0]
        count = 0
        # searches every row in the mask from bottom to top
        # when more than 19 pixels of the object in a row are found, the row number is saved
        # if the next 30 rows have also at least 20 pixels from the object in them, the second row number is saved
        # both row numbers define the area where the lowest part of the object is
        for i in range(y-1, -1, -1):
            if count >= int(y/100):
                area[0] = i
                break
            line = map2[i, :]
            if len(line[line != 0]) >= int(y/150):
                if area[1] == 0:
                    area[1] = i
                    count = 1
                else:
                    count = count + 1
            else:
                area[1] = 0
                count = 0
        # focus depth from minimum depth of the object
        segment = map2[area[0]:area[1], :]
        focus = int(np.mean(segment[segment != 0]))

        # focus depth from mean depth of entire object
        # focus = np.mean(map2[map2 > 0])

        # giving the entire object the same depth as the focus/mean depth
        d_map[map2 > 0] = focus
    return d_map, focus

