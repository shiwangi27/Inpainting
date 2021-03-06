import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *

# n_epochs = 10000
# n_epochs = 100
n_epochs = 10
learning_rate_val = 0.0003
weight_decay_rate =  0.00001
momentum = 0.9
batch_size = 256
lambda_recon = 0.9
lambda_adv = 0.1

border_size = 7
giving_size = 64

trainset_path = '../data/imagenet_out_trainset.pickle'
testset_path  = '../data/imagenet_out_testset.pickle'
dataset_path = '../data/train2014'
model_path = '../models/imagenet_out/'
result_path= '../results/imagenet_out/'
pretrained_model_path = '../models/imagenet_out/model-0'

if not os.path.exists(model_path):
    os.makedirs( model_path )

if not os.path.exists(result_path):
    os.makedirs( result_path )

if not os.path.exists( trainset_path ) or not os.path.exists( testset_path ):
    imagenet_images = []
    for dir, _, _, in os.walk(dataset_path):
        imagenet_images.extend( glob( os.path.join(dir, '*.jpg')))

    imagenet_images = np.hstack(imagenet_images)

    trainset = pd.DataFrame({'image_path':imagenet_images[:int(len(imagenet_images)*0.9)]})
    testset = pd.DataFrame({'image_path':imagenet_images[int(len(imagenet_images)*0.9):]})

    trainset.to_pickle( trainset_path )
    testset.to_pickle( testset_path )
else:
    trainset = pd.read_pickle( trainset_path )
    testset = pd.read_pickle( testset_path )

testset.index = range(len(testset))
# print 'len(testset):', len(testset)
testset = testset.ix[np.random.permutation(len(testset))]
is_train = tf.placeholder( tf.bool )

learning_rate = tf.placeholder( tf.float32, [])
images_tf = tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images")
print 'image_tf:', images_tf
labels_D = tf.concat( 0, [tf.ones([batch_size]), tf.zeros([batch_size])] )
print 'labels_D:', labels_D
# print labels_D[0]
# print labels_D[499]
# print labels_D[500]
# print labels_D[999]
labels_G = tf.ones([batch_size])
print 'labels_G:', labels_G
images_giving = tf.placeholder( tf.float32, [batch_size, giving_size, giving_size, 3], name='images_giving')
print 'imags_giving:', images_giving

model = Model()

bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction = model.build_reconstruction(images_tf, is_train)
print 'reconstruction:', reconstruction
adversarial_pos = model.build_adversarial(images_giving, is_train)
adversarial_neg = model.build_adversarial(reconstruction, is_train, reuse=True)
adversarial_all = tf.concat(0, [adversarial_pos, adversarial_neg])

# Applying bigger loss for overlapping region
mask_overlap = tf.pad(tf.ones([giving_size - 2*border_size, giving_size - 2*border_size]), [[border_size,border_size], [border_size,border_size]])
mask_overlap = tf.reshape(mask_overlap, [giving_size, giving_size, 1])
mask_overlap = tf.concat(2, [mask_overlap]*3)
mask_border = 1 - mask_overlap

loss_recon_ori = tf.square( images_giving - reconstruction )
loss_recon_border = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_border, [1,2,3]))) / 10.  # Loss for non-overlapping region
loss_recon_overlap = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_overlap, [1,2,3]))) # Loss for overlapping region
loss_recon = loss_recon_border + loss_recon_overlap

loss_adv_D = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_all, labels_D))
loss_adv_G = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_neg, labels_G))

loss_G = loss_adv_G * lambda_adv + loss_recon * lambda_recon
loss_D = loss_adv_D * lambda_adv

var_G = filter( lambda x: x.name.startswith('GEN'), tf.trainable_variables())
var_D = filter( lambda x: x.name.startswith('DIS'), tf.trainable_variables())

print 'var_G:', var_G
for g in var_G:
    print g.name
print 'var_D:', var_D
for d in var_D:
    print d.name

W_G = filter(lambda x: x.name.endswith('W:0'), var_G)
W_D = filter(lambda x: x.name.endswith('W:0'), var_D)

# print 'W_G:', W_G
# for g in W_G:
#     print g.name
# print 'W_D:', W_D
# for d in W_D:
#     print d.name

loss_G += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_G)))
loss_D += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_D)))

# print 'loss_G:', loss_G
# print 'loss_D:', loss_D

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

optimizer_G = tf.train.AdamOptimizer( learning_rate=learning_rate )

# print 'optimizer_G:', optimizer_G

grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
# for x in xrange(len(grads_vars_G)):
#     print x, ':', grads_vars_G[x][0], grads_vars_G[x][1].name
# print type(grads_vars_G[0])
# print type(grads_vars_G[2])
# grads_vars_G = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G)
for gv in grads_vars_G:
    if gv[0] is not None:
        gv = [tf.clip_by_value(gv[0], -10., 10.), gv[1]]
# print ''
# for x in xrange(len(grads_vars_G)):
#     print x, ':', grads_vars_G[x][0], grads_vars_G[x][1].name
# print type(grads_vars_G[0])
# print type(grads_vars_G[2])
train_op_G = optimizer_G.apply_gradients( grads_vars_G )

optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate )
grads_vars_D = optimizer_D.compute_gradients( loss_D, var_list=var_D )
# grads_vars_D = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D)
for gv in grads_vars_D:
    if gv[0] is not None:
        gv = [tf.clip_by_value(gv[0], -10., 10.), gv[1]]
train_op_D = optimizer_D.apply_gradients( grads_vars_D )

saver = tf.train.Saver(max_to_keep=100)

tf.initialize_all_variables().run()

if pretrained_model_path is not None and os.path.exists( pretrained_model_path ):
    saver.restore( sess, pretrained_model_path )

iters = 0

loss_D_val = 0.
loss_G_val = 0.

for epoch in range(n_epochs):
    trainset.index = range(len(trainset))
    trainset = trainset.ix[np.random.permutation(len(trainset))]

    for start,end in zip(
            range(0, len(trainset), batch_size),
            range(batch_size, len(trainset), batch_size)):

        image_paths = trainset[start:end]['image_path'].values
        images_ori = map(lambda x: load_image( x ), image_paths)
        is_none = np.sum(map(lambda x: x is None, images_ori))
        if is_none > 0: continue

        images_crops = map(lambda x: crop_random(x), images_ori)
        ##print 'in main: images_crops:', images_crops
        images, crops,_,_ = zip(*images_crops)

        # Printing activations every 10 iterations
        if iters % 20 == 0:
            test_image_paths = testset[:batch_size]['image_path'].values
            test_images_ori = map(lambda x: load_image(x), test_image_paths)

            test_images_crop = map(lambda x: crop_random(x, x=32, y=32), test_images_ori)
            test_images, test_crops, xs, ys = zip(*test_images_crop)
            # test_images are 128x128 (50x50 original, rest placeholder)
            # test_crops are 64*64 original

            reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
                    [reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, loss_G, loss_D],
                    feed_dict={
                        images_tf: test_images,
                        images_giving: test_crops,
                        is_train: False
                        })

            # print 'reconstruction_vals shape:', reconstruction_vals.shape

            # Generate result every 500 iterations
            if iters % 100 == 0:
                ii = 0
                for rec_val, img, x, y in zip(reconstruction_vals, test_images, xs, ys):
                    # print 'rec_val type:', type(rec_val)
                    rec_border = (255. * (rec_val+1)/2.).astype(int)
                    rec_con = (255. * (img+1)/2.).astype(int)

                    # rec_con[y:y+64, x:x+64] = rec_hid
                    rec_con = rec_con[y:y+64, x:x+64]
                    rec_con[:7, :57] = rec_border[:7, :57]
                    rec_con[:57, 57:] = rec_border[:57, 57:]
                    rec_con[57:, 7:] = rec_border[57:, 7:]
                    rec_con[7:, :7] = rec_border[7:, :7]
                    cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.'+str(int(iters/100))+'.jpg'), rec_con)
                    ii += 1
                    if ii > 50: break

                if iters == 0:
                    ii = 0
                    for test_image in test_images_ori:
                        test_image = (255. * (test_image+1)/2.).astype(int)
                        # test_image[32:32+64,32:32+64] = 0
                        test_image[:32,:32+64] = 0
                        test_image[:32+64,32+64:] = 0
                        test_image[32+64:,32:] = 0
                        test_image[32:,:32] = 0
                        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.ori.jpg'), test_image)
                        ii += 1
                        if ii > 50: break

            print "========================================================================"
            print bn1_val.max(), bn1_val.min()
            print bn2_val.max(), bn2_val.min()
            print bn3_val.max(), bn3_val.min()
            print bn4_val.max(), bn4_val.min()
            print bn5_val.max(), bn5_val.min()
            print bn6_val.max(), bn6_val.min()
            print debn4_val.max(), debn4_val.min()
            print debn3_val.max(), debn3_val.min()
            print debn2_val.max(), debn2_val.min()
            print debn1_val.max(), debn1_val.min()
            print recon_ori_vals.max(), recon_ori_vals.min()
            print reconstruction_vals.max(), reconstruction_vals.min()
            print loss_G_val, loss_D_val
            print "========================================================================="

            if np.isnan(reconstruction_vals.min() ) or np.isnan(reconstruction_vals.max()):
                print "NaN detected!!"
                ipdb.set_trace()

        # Generative Part is updated every iteration
        _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val = sess.run(
                [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1],
                feed_dict={
                    images_tf: images,
                    images_giving: crops,
                    learning_rate: learning_rate_val,
                    is_train: True
                    })

        if iters % 10 == 0:
            _, loss_D_val, adv_pos_val, adv_neg_val = sess.run(
                    [train_op_D, loss_D, adversarial_pos, adversarial_neg],
                    feed_dict={
                        images_tf: images,
                        images_giving: crops,
                        learning_rate: learning_rate_val,
                        is_train: True
                            })

            print "Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_val, "Gen ADV Loss:", loss_adv_G_val,  "Dis Loss:", loss_D_val, "||||", adv_pos_val.mean(), adv_neg_val.min(), adv_neg_val.max()

        iters += 1


    saver.save(sess, model_path + 'model', global_step=epoch)
    learning_rate_val *= 0.99


