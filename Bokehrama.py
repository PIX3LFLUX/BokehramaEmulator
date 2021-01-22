# **********************************************************************************************************************
# Jeff Manderscheid, Matriculation Number: 60037
# Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
# Bokehrama Emulation V1.0, 22.09.2020
# **********************************************************************************************************************

import cv2
from fileHandler import save_images, read_masks_maps, save_bokehramas
from utility import create_seg_mask, obj_focus, adj_depth, blender
from opt_segmentation import enhance_seg
from bokeh import create_bokeh, create_bokeh_disks


# Can create multiple Bokehramas. Multiple Segmentation must be done before.
def multi_bokehrama(images, exp=3, filtersize=21):
    masks, d_maps = read_masks_maps(images)
    im_blender = []
    cut_seg = []
    for i, image in enumerate(images):
        # Reading in an image
        im = cv2.imread(image, cv2.IMREAD_COLOR)
        x, y, _ = im.shape

        # Converting colors
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        masks[i] = cv2.cvtColor(masks[i], cv2.COLOR_BGR2GRAY)

        # creating mask of object
        masks[i] = create_seg_mask(masks[i], True)
        # mask = create_mask(mask)

        # mask enhancement
        masks[i] = enhance_seg(im, masks[i], True, True)

        # determine depth of object in focus
        (d_maps[i], focus) = obj_focus(d_maps[i], masks[i])

        # Adjusting the depth map with the focal depth
        d_maps[i] = adj_depth(d_maps[i], focus)

        # Gradient blur with blender
        # k_size = int(y/500)
        im_blur = create_bokeh(im, filtersize=filtersize)
        im_blender.append(blender(im, im_blur, d_maps[i], exponent=exp))
        im_blender[i] = cv2.cvtColor(im_blender[i], cv2.COLOR_RGB2BGR)
        # test zone

    # save the Bokehrama
    save_images(cut_seg, images, "canny", folder=True)
    save_images(masks, images, 'mask', folder=True)
    save_bokehramas(im_blender, images)
