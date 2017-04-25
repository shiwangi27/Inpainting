import skimage.io
import skimage.transform
from PIL import ImageFile
import os
import ipdb

import numpy as np

#def load_image( path, height=128, width=128 ):
def load_image( path, pre_height=146, pre_width=146, height=128, width=128 ):

    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    # crop the center of the image
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

    # make the values in range (-1, 1)
    return (resized_img * 2)-1 #(resized_img - 127.5)/127.5

def crop_random(image_ori, width=64,height=64, x=None, y=None, border=7):
    if image_ori is None: return None
    random_y = np.random.randint(0,height) if x is None else x
    random_x = np.random.randint(0,width) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    # image[random_y + border:random_y+height - border, random_x + border:random_x+width - border, 0] = 2*117. / 255. - 1.
    # image[random_y + border:random_y+height - border, random_x + border:random_x+width - border, 1] = 2*104. / 255. - 1.
    # image[random_y + border:random_y+height - border, random_x + border:random_x+width - border, 2] = 2*123. / 255. - 1.
    image[:random_y + border, :random_x + width - border, 0] = 2*117. / 255. - 1.
    image[:random_y + border, :random_x + width - border, 1] = 2*104. / 255. - 1.
    image[:random_y + border, :random_x + width - border, 2] = 2*123. / 255. - 1.
    image[:random_y + height - border, random_x + width - border:, 0] = 2*117. / 255. - 1.
    image[:random_y + height - border, random_x + width - border:, 1] = 2*104. / 255. - 1.
    image[:random_y + height - border, random_x + width - border:, 2] = 2*123. / 255. - 1.
    image[random_y + height - border:, random_x + border:, 0] = 2*117. / 255. - 1.
    image[random_y + height - border:, random_x + border:, 1] = 2*104. / 255. - 1.
    image[random_y + height - border:, random_x + border:, 2] = 2*123. / 255. - 1.
    image[random_y + border:, :random_x + border, 0] = 2*117. / 255. - 1.
    image[random_y + border:, :random_x + border, 1] = 2*104. / 255. - 1.
    image[random_y + border:, :random_x + border, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y
