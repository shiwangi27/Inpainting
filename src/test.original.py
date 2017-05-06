#import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *

n_epochs = 10000
learning_rate_val = 0.001
weight_decay_rate = 0.00001
momentum = 0.9
batch_size = 128
lambda_recon = 0.999
lambda_adv = 0.001

overlap_size = 7
giving_size = 64

testset_path  = '../data/imagenet_out_testset.pickle'
result_path= '../results/imagenet_test/'
pretrained_model_path = '../models/imagenet_out/model-150'
testset = pd.read_pickle( testset_path )

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
saver.restore( sess, pretrained_model_path )

iters = 0

ii = 0
for start,end in zip(
        range(0, len(testset), batch_size),
        range(batch_size, len(testset), batch_size)):

    print("iteration................", ii )
    test_image_paths = testset[:batch_size]['image_path'].values
    test_images_ori = map(lambda x: load_image(x), test_image_paths)

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
        
        img_rgb = (255. * (img + 1)/2.).astype(int)
        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.ori.jpg'), img_rgb)
        cv2.imwrite( os.path.join(result_path, 'img_ori'+str(ii)+'.'+str(int(iters/1000))+'.jpg'), rec_con)
        ii += 1
        if ii > 30: break


