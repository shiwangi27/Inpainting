#!/usr/bin/env python

import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *

from argparse import ArgumentParser

# def outpaint(data_in, paths_out, pretrained_model_path, device_t='/gpu:0', batch_size=4):
def outpaint(test_image_path, path_out, checkpoint_dir):
    batch_size = 1

    overlap_size = 7
    giving_size = 64

    is_train = tf.placeholder( tf.bool )
    images_tf = tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images")
    images_giving = tf.placeholder( tf.float32, [batch_size, giving_size, giving_size, 3], name='images_giving')

    model = Model()

    #reconstruction = model.build_reconstruction(images_tf, is_train)
    bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction = model.build_reconstruction(images_tf, is_train)

    # Applying bigger loss for overlapping region
    sess = tf.InteractiveSession()
    #
    saver = tf.train.Saver(max_to_keep=100)

    tf.initialize_all_variables().run()

    if os.path.isdir(checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")
    else:
        saver.restore(sess, checkpoint_dir)

    test_images_ori = map(lambda x: load_image(x), [test_image_path])

    test_images_crop = map(lambda x: crop_random(x, x=32, y=32), test_images_ori)
    test_images, test_crops, xs,ys = zip(*test_images_crop)

    reconstruction_vals = sess.run(
            reconstruction,
            feed_dict={
                images_tf: test_images,
                images_giving: test_crops,
                is_train: False
                })

    for rec_val,img,x,y in zip(reconstruction_vals,test_images, xs, ys):
        rec_border = (255. * (rec_val+1)/2.).astype(int)
        rec_con = (255. * (img+1)/2.).astype(int)

        rec_con = rec_con[y:y+64, x:x+64]
        rec_con[:7, :57] = rec_border[:7, :57]
        rec_con[:57, 57:] = rec_border[:57, 57:]
        rec_con[57:, 7:] = rec_border[57:, 7:]
        rec_con[7:, :7] = rec_border[7:, :7]

        # img_rgb = (255. * (img + 1)/2.).astype(int)
        # save_img(path_out, img_rgb)
        save_img(path_out, rec_con)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='.ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='file to outpaint',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    # parser.add_argument('--device', type=str,
    #                     dest='device',help='device to perform compute on',
    #                     metavar='DEVICE', default=DEVICE)

    # parser.add_argument('--batch-size', type=int,
    #                     dest='batch_size',help='batch size for feedforwarding',
    #                     metavar='BATCH_SIZE', default=BATCH_SIZE)

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        # assert opts.batch_size > 0

def main():

    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path
    else:
        out_path = os.path.join(opts.out_path,os.path.basename(opts.in_path))

    outpaint(opts.in_path, out_path, opts.checkpoint_dir)

if __name__ == '__main__':
    main()


