# **********************************************************************************************************************
# Jeff Manderscheid, Matriculation Number: 60037
# Faculty of Electrical Engineering and Information Technology, Hochschule Karlsruhe
# Bokehrama Emulation V1.0, 22.09.2020
# **********************************************************************************************************************

import os
import cv2


# reads all the images in th given directory, and returns the names with the path as an array
def read_images(args):
    images = []
    # reads every file in the given directory
    for file in os.listdir(args.path):
        # checks for files of the type png, jpg or jpeg
        if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.jpeg':
            images.append(args.path+'/'+file)
    # returns all images found as an array
    return images


# gives back the directory from a array of image paths
def give_images_path(images):
    return os.path.dirname(images[0])


# reads the masks and maps from the files for further processing
def read_masks_maps(images):
    masks = []
    d_maps = []
    # reads every file in the directory
    for image in images:
        # checks for segmentation masks
        masks.append(cv2.imread((os.path.dirname(image) + '/masks_maps/' + os.path.splitext(os.path.basename(image))[0]
                                + '_seg.png'), cv2.IMREAD_COLOR))
        # checks for depth maps
        d_maps.append(cv2.imread((os.path.dirname(image) + '/masks_maps/' + os.path.splitext(os.path.basename(image))[0]
                                + '_disp.jpeg'), cv2.IMREAD_GRAYSCALE))
    # return both masks and maps as arrays
    return masks, d_maps


# can save an array of images into multiple files, using the names of the original images and a name addition
# enabling the folder means the images will be saved into a subfolder /masks_maps/
# creates subfolder if it does not exist
def save_images(images, paths, name, folder=False):
    new_paths = []
    # saves into subfolder
    if folder:
        # checks if the subfolder exists and creates one if it does`t
        if not os.path.exists(os.path.dirname(paths[0]) + '/masks_maps/'):
            os.mkdir(os.path.dirname(paths[0]) + '/masks_maps/')
        # goes through the array of images while counting
        for i, image in enumerate(images):
            # adds to every original path the subfolder directory
            new_paths.append(os.path.dirname(paths[i]) + '/masks_maps/' + os.path.splitext(os.path.basename(paths[i]))[0]
                         + '_' + name + '.jpg')
            cv2.imwrite(new_paths[i], image)
    # saves into the same folder where the original images were
    else:
        for i, image in enumerate(images):
            new_paths.append(os.path.splitext(paths[i])[0]+'_'+name+'.jpg')
            cv2.imwrite(new_paths[i], image)
    return new_paths


# can save an array of bokehramas into multiple files
# if the subfolder /bokehramas/ does not exist it gets created
def save_bokehramas(images, paths):
    new_paths = []
    # checks if the subfolder exists and creates one if it does`t
    if not os.path.exists(os.path.dirname(paths[0]) + '/bokehramas/'):
        os.mkdir(os.path.dirname(paths[0]) + '/bokehramas/')
    for i, image in enumerate(images):
        # adds to every original path the subfolder directory
        new_paths.append(os.path.dirname(paths[i]) + '/bokehramas/' + os.path.splitext(os.path.basename(paths[i]))[0]
                     + '_bokehrama.jpg')
        cv2.imwrite(new_paths[i], image)
