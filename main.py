# **********************************************************************************************************************
# Jeff Manderscheid, Matriculation Number: 60037
# Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
# Bokehrama Emulation V1.0, 22.09.2020
# **********************************************************************************************************************

from Bokehrama import multi_bokehrama
from segment import multi_seg
from fileHandler import read_images, give_images_path
from depth_estimation import estimate
from utility import parse_args


if __name__ == '__main__':
    args = parse_args()
    im = read_images(args)
    im_path = give_images_path(im)
    print(str(len(im)) + ' image(s) found...\nProcessing Segmentation...')
    multi_seg(im)
    print('Segmentation done...\nProcessing Depth Estimation...')
    estimate(im_path)
    print('Depth estimation done...\nCreating the Bokehramas...')
    # The exponent changes the DOF, the higher the value the shallower the DOF
    # The filtersize is the size of the Gaussian kernel to create the blur, number must end with a 1 (e.g. 11,21,31,...)
    multi_bokehrama(im, exp=2, filtersize=21)
    print('Bokehrama Creation done...')
