# **********************************************************************************************************************
# Jeff Manderscheid, Matriculation Number: 60037
# Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
# Bokehrama Emulation V1.0, 22.09.2020
# **********************************************************************************************************************

from keras_segmentation.pretrained import pspnet_50_ADE_20K
import os


# Semantic Segmentation for an undefined number of images. It needs the names as an array.
def multi_seg(images):
    model = pspnet_50_ADE_20K()

    for im in images:
        if not os.path.exists(os.path.dirname(im) + '/masks_maps/'):
            os.mkdir(os.path.dirname(im) + '/masks_maps/')
        out = model.predict_segmentation(
            inp=im,
            out_fname=(os.path.dirname(im)+'/masks_maps/'+os.path.splitext(os.path.basename(im))[0]+'_seg.png')
        )
