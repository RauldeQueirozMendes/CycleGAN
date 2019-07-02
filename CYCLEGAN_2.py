from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Do other imports now...
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

print("oi")

import scipy
import os
import cv2
import numpy as np
import glob
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import time
import warnings
import argparse
from scipy.interpolate import LinearNDInterpolator
from collections import Counter
from tqdm import tqdm
from collections import Counter
from scipy.interpolate import LinearNDInterpolator
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import apply_brightness_shift,apply_channel_shift,img_to_array,load_img,apply_affine_transform
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default

plt.ion()

def build_generator_BA():
    """U-Net Generator"""

    # Number of filters in the first layer of G and D
    gf = 64

    def conv2d1(layer_input,filters,f_size=3):
        """Layers used during downsampling"""
        d = tf.keras.layers.Conv2D(filters,kernel_size=f_size,strides=1,padding='same')(layer_input)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.BatchNormalization()(d)
        return d

    def conv2d(layer_input,filters,f_size=3):
        """Layers used during downsampling"""
        d = tf.keras.layers.Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.BatchNormalization()(d)
        return d

    def deconv2d(layer_input,skip_input,filters,f_size=2,dropout_rate=0):
        """Layers used during upsampling"""
        u = tf.keras.layers.Conv2DTranspose(filters,kernel_size=f_size,strides=2,padding='same',activation='relu')(layer_input)
        if dropout_rate:
            u = tf.keras.layers.Dropout(dropout_rate)(u)
        u = tf.keras.layers.BatchNormalization()(u)
        u = tf.keras.layers.Concatenate()([u,skip_input])
        return u

    # Image input
    d0 = tf.keras.Input(shape=depth_shape)

    # Downsampling
    d1 = conv2d1(d0,gf)
    d1 = conv2d1(d1,gf)
    d2 = conv2d(d1,gf * 2)
    d2 = conv2d1(d2,gf * 2)
    d3 = conv2d(d2,gf * 4)
    d3 = conv2d1(d3,gf * 4)
    d4 = conv2d(d3,gf * 8)
    d4 = conv2d1(d4,gf * 8)
    d5 = conv2d(d4,gf * 16)
    d5 = conv2d1(d5,gf * 16)

    # Upsampling
    u0 = deconv2d(d5,d4,gf * 8)
    u0 = tf.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same',activation='relu')(u0)
    u0 = tf.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same',activation='relu')(u0)
    u1 = deconv2d(u0,d3,gf * 4)
    u1 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same',activation='relu')(u1)
    u1 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same',activation='relu')(u1)
    u2 = deconv2d(u1,d2,gf * 2)
    u2 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same',activation='relu')(u2)
    u2 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same',activation='relu')(u2)
    u3 = deconv2d(u2,d1,gf)
    u3 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(2,kernel_size=3,strides=1,padding='same',activation='relu')(u3)

    output_img = tf.keras.layers.Conv2D(channels,1,activation='sigmoid')(u3)

    return tf.keras.Model(d0,output_img)

def build_generator_AB():
    """U-Net Generator"""

    # Number of filters in the first layer of G and D
    gf = 64

    def conv2d1(layer_input,filters,f_size=3):
        """Layers used during downsampling"""
        d = tf.keras.layers.Conv2D(filters,kernel_size=f_size,strides=1,padding='same')(layer_input)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.BatchNormalization()(d)
        return d

    def conv2d(layer_input,filters,f_size=3):
        """Layers used during downsampling"""
        d = tf.keras.layers.Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.BatchNormalization()(d)
        return d

    def deconv2d(layer_input,skip_input,filters,f_size=2,dropout_rate=0):
        """Layers used during upsampling"""
        u = tf.keras.layers.Conv2DTranspose(filters,kernel_size=f_size,strides=2,padding='same',activation='relu')(layer_input)
        if dropout_rate:
            u = tf.keras.layers.Dropout(dropout_rate)(u)
        u = tf.keras.layers.BatchNormalization()(u)
        u = tf.keras.layers.Concatenate()([u,skip_input])
        return u

    # Image input
    d0 = tf.keras.Input(shape=img_shape)

    # Downsampling
    d1 = conv2d1(d0,gf)
    d1 = conv2d1(d1,gf)
    d2 = conv2d(d1,gf * 2)
    d2 = conv2d1(d2,gf * 2)
    d3 = conv2d(d2,gf * 4)
    d3 = conv2d1(d3,gf * 4)
    d4 = conv2d(d3,gf * 8)
    d4 = conv2d1(d4,gf * 8)
    d5 = conv2d(d4,gf * 16)
    d5 = conv2d1(d5,gf * 16)

    # Upsampling
    u0 = deconv2d(d5,d4,gf * 8)
    u0 = tf.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same',activation='relu')(u0)
    u0 = tf.keras.layers.Conv2D(512,kernel_size=3,strides=1,padding='same',activation='relu')(u0)
    u1 = deconv2d(u0,d3,gf * 4)
    u1 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same',activation='relu')(u1)
    u1 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same',activation='relu')(u1)
    u2 = deconv2d(u1,d2,gf * 2)
    u2 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same',activation='relu')(u2)
    u2 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same',activation='relu')(u2)
    u3 = deconv2d(u2,d1,gf)
    u3 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(2,kernel_size=3,strides=1,padding='same',activation='relu')(u3)


    output_img = tf.keras.layers.Conv2D(channels_depth,1,activation='sigmoid')(u3)

    return tf.keras.Model(d0,output_img)

def build_discriminator_A():

    # Number of filters in the first layer of G and D
    df = 64

    def d_layer(layer_input,filters,f_size=4,normalization=True):
        """Discriminator layer"""
        d = tf.keras.layers.Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
        d = tf.keras.layers.ReLU()(d)
        if normalization:
            d = tf.keras.layers.BatchNormalization()(d)
        return d

    img = tf.keras.Input(shape=img_shape)

    d1 = d_layer(img,df,normalization=False)
    d2 = d_layer(d1,df * 2)
    d3 = d_layer(d2,df * 4)
    d4 = d_layer(d3,df * 8)
    d5 = d_layer(d4,df * 16)

    validity = tf.keras.layers.Conv2D(1,kernel_size=4,strides=1,padding='same')(d5)

    return tf.keras.Model(img,validity)

def build_discriminator_B():

    # Number of filters in the first layer of G and D
    df = 64

    def d_layer(layer_input,filters,f_size=4,normalization=True):
        """Discriminator layer"""
        d = tf.keras.layers.Conv2D(filters,kernel_size=f_size,strides=2,padding='same')(layer_input)
        d = tf.keras.layers.ReLU()(d)
        if normalization:
            d = tf.keras.layers.BatchNormalization()(d)
        return d

    img = tf.keras.Input(shape=depth_shape)

    d1 = d_layer(img,df,normalization=False)
    d2 = d_layer(d1,df * 2)
    d3 = d_layer(d2,df * 4)
    d4 = d_layer(d3,df * 8)
    d5 = d_layer(d4,df * 16)

    validity = tf.keras.layers.Conv2D(1,kernel_size=4,strides=1,padding='same')(d5)

    return tf.keras.Model(img,validity)


# Input shape
img_rows = 128
img_cols = 416
channels = 3
channels_depth = 1
img_shape = (img_rows,img_cols,channels)
depth_shape = (img_rows,img_cols,channels_depth)

# Calculate output shape of D (PatchGAN)
patch = int(img_rows / 2 ** 5)
patch2 = int(img_cols / 2 ** 5)
disc_patch = (patch,patch2,1)

# Loss weights
lambda_cycle = 10.0  # Cycle-consistency loss
# lambda_id = 0.1 * lambda_cycle  # Identity loss

optimizer = tf.keras.optimizers.Adam(0.0001,0.5)

# Build and compile the discriminators
d_A = build_discriminator_A()
d_A.summary()
d_B = build_discriminator_B()
d_B.summary()

#--------------------------------------------------

# d_A.load_weights('d_A_weights.h5')

#--------------------------------------------------

#--------------------------------------------------

# d_B.load_weights('d_B_weights.h5')

#--------------------------------------------------

# ===========
#  Mask     #
# ===========
def tf_mask_out_invalid_pixels(tf_pred, tf_true):
    # Identify Pixels to be masked out.
    tf_idx = tf.where(tf_true > 0.0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

    # Mask Out Pixels without depth values
    tf_valid_pred = tf.gather_nd(tf_pred, tf_idx)
    tf_valid_true = tf.gather_nd(tf_true, tf_idx)

    return tf_valid_pred, tf_valid_true

# -------------------- #
#  Mean Squared Error  #
# -------------------- #
def tf_mse_loss(y_true,y_pred):

    valid_pixels = True

    # Mask Out
    if valid_pixels:
        y_pred, y_true = tf_mask_out_invalid_pixels(y_pred, y_true)

    # npixels value depends on valid_pixels flag:
    # npixels = (batchSize*height*width) OR npixels = number of valid pixels
    tf_npixels = tf.cast(tf.size(y_true), tf.float32)

    # Loss
    mse = tf.div(tf.reduce_sum(tf.square(y_true - y_pred)), tf_npixels)

    return mse

# -------------------- #
#  Mean Absolute Error #
# -------------------- #
def tf_mae_loss(y_true,y_pred):

    valid_pixels = True

    # Mask Out
    if valid_pixels:
        y_pred, y_true = tf_mask_out_invalid_pixels(y_pred, y_true)

    # npixels value depends on valid_pixels flag:
    # npixels = (batchSize*height*width) OR npixels = number of valid pixels
    tf_npixels = tf.cast(tf.size(y_true), tf.float32)

    # Loss
    mae = tf.div(tf.reduce_sum(tf.abs(y_true - y_pred)), tf_npixels)

    return mae

# ------- #
#  BerHu  #
# ------- #
def tf_berhu_loss(y_true, y_pred):
    valid_pixels = True

    # C Constant Calculation
    tf_abs_error = tf.abs(y_pred - y_true, name='abs_error')
    tf_c = tf.multiply(tf.constant(0.2), tf.reduce_max(tf_abs_error))  # Consider All Pixels!

    # Mask Out
    if valid_pixels:
        # Overwrites the 'y' and 'y_' tensors!
        y_pred, y_true = tf_mask_out_invalid_pixels(y_pred, y_true)

        # Overwrites the previous tensor, so now considers only the Valid Pixels!
        tf_abs_error = tf.abs(y_pred - y_true, name='abs_error')

    # Loss
    tf_berhu_loss = tf.where(tf_abs_error <= tf_c, tf_abs_error,
                             tf.div((tf.square(tf_abs_error) + tf.square(tf_c)), tf.multiply(tf.constant(2.0), tf_c)))

    tf_loss = tf.reduce_sum(tf_berhu_loss)

    return tf_loss


d_A.compile(loss=tf_berhu_loss,
                 optimizer=optimizer,
                 metrics=['accuracy'])
d_B.compile(loss=tf_berhu_loss,
                 optimizer=optimizer,
                 metrics=['accuracy'])

# -------------------------
# Construct Computational
#   Graph of Generators
# -------------------------

# Build the generators
g_AB = build_generator_AB()
g_AB.summary()
g_BA = build_generator_BA()
g_BA.summary()

#--------------------------------------------------

# g_AB.load_weights('g_AB_weights.h5')

#--------------------------------------------------

#--------------------------------------------------

# g_BA.load_weights('g_BA_weights.h5')

#--------------------------------------------------

# Input images from both domains
img_A = tf.keras.Input(shape=img_shape)
img_B = tf.keras.Input(shape=depth_shape)

# Translate images to the other domain
fake_B = g_AB(img_A)
fake_A = g_BA(img_B)
# Translate images back to original domain
reconstr_A = g_BA(fake_B)
reconstr_B = g_AB(fake_A)
# Identity mapping of images
# img_A_id = g_BA(img_A)
# img_B_id = g_AB(img_B)

# For the combined model we will only train the generators
d_A.trainable = False
d_B.trainable = False

# Discriminators determines validity of translated images
valid_A = d_A(fake_A)
valid_B = d_B(fake_B)

# Combined model trains generators to fool discriminators
combined = tf.keras.Model(inputs=[img_A,img_B],
                        outputs=[valid_A,valid_B,reconstr_A,reconstr_B])
combined.summary()

#--------------------------------------------------

# combined.load_weights('combined_weights.h5')

#--------------------------------------------------

combined.compile(loss=[tf_berhu_loss,tf_berhu_loss,tf_mae_loss,tf_mae_loss],
                loss_weights=[1,1,lambda_cycle,lambda_cycle],
                optimizer=optimizer)

start_time = datetime.datetime.now()

batch_size=4

# Adversarial loss ground truths
valid = np.ones((batch_size,) + disc_patch)
fake = np.zeros((batch_size,) + disc_patch)

# print(valid)
# print(fake)

epochs=100000

sample_interval=1

def load_and_scale_image(filepath):
    image_input = img_to_array(load_img(filepath, target_size=(img_rows,img_cols), interpolation='lanczos'))
    image_input = image_input.astype(np.float32)
    image_input = np.expand_dims(image_input,axis=0)
    return image_input/255.0

def load_and_scale_depth(filepath):
    image_input = img_to_array(load_img(filepath, grayscale=True, color_mode='grayscale', target_size=(img_rows,img_cols), interpolation='lanczos'))/3.0
    image_input = image_input.astype(np.float32)
    image_input = np.expand_dims(image_input,axis=0)
    return image_input/90.0


if not (os.path.exists('kitti_continuous_train (2).txt') and os.path.exists('kitti_continuous_test (2).txt')):

    timer1 = -time.time()

    bad_words = ['image_03',
                 '2011_09_28_drive_0053_sync',
                 '2011_09_28_drive_0054_sync',
                 '2011_09_28_drive_0057_sync',
                 '2011_09_28_drive_0065_sync',
                 '2011_09_28_drive_0066_sync',
                 '2011_09_28_drive_0068_sync',
                 '2011_09_28_drive_0070_sync',
                 '2011_09_28_drive_0071_sync',
                 '2011_09_28_drive_0075_sync',
                 '2011_09_28_drive_0077_sync',
                 '2011_09_28_drive_0078_sync',
                 '2011_09_28_drive_0080_sync',
                 '2011_09_28_drive_0082_sync',
                 '2011_09_28_drive_0086_sync',
                 '2011_09_28_drive_0087_sync',
                 '2011_09_28_drive_0089_sync',
                 '2011_09_28_drive_0090_sync',
                 '2011_09_28_drive_0094_sync',
                 '2011_09_28_drive_0095_sync',
                 '2011_09_28_drive_0096_sync',
                 '2011_09_28_drive_0098_sync',
                 '2011_09_28_drive_0100_sync',
                 '2011_09_28_drive_0102_sync',
                 '2011_09_28_drive_0103_sync',
                 '2011_09_28_drive_0104_sync',
                 '2011_09_28_drive_0106_sync',
                 '2011_09_28_drive_0108_sync',
                 '2011_09_28_drive_0110_sync',
                 '2011_09_28_drive_0113_sync',
                 '2011_09_28_drive_0117_sync',
                 '2011_09_28_drive_0119_sync',
                 '2011_09_28_drive_0121_sync',
                 '2011_09_28_drive_0122_sync',
                 '2011_09_28_drive_0125_sync',
                 '2011_09_28_drive_0126_sync',
                 '2011_09_28_drive_0128_sync',
                 '2011_09_28_drive_0132_sync',
                 '2011_09_28_drive_0134_sync',
                 '2011_09_28_drive_0135_sync',
                 '2011_09_28_drive_0136_sync',
                 '2011_09_28_drive_0138_sync',
                 '2011_09_28_drive_0141_sync',
                 '2011_09_28_drive_0143_sync',
                 '2011_09_28_drive_0145_sync',
                 '2011_09_28_drive_0146_sync',
                 '2011_09_28_drive_0149_sync',
                 '2011_09_28_drive_0153_sync',
                 '2011_09_28_drive_0154_sync',
                 '2011_09_28_drive_0155_sync',
                 '2011_09_28_drive_0156_sync',
                 '2011_09_28_drive_0160_sync',
                 '2011_09_28_drive_0161_sync',
                 '2011_09_28_drive_0162_sync',
                 '2011_09_28_drive_0165_sync',
                 '2011_09_28_drive_0166_sync',
                 '2011_09_28_drive_0167_sync',
                 '2011_09_28_drive_0168_sync',
                 '2011_09_28_drive_0171_sync',
                 '2011_09_28_drive_0174_sync',
                 '2011_09_28_drive_0177_sync',
                 '2011_09_28_drive_0179_sync',
                 '2011_09_28_drive_0183_sync',
                 '2011_09_28_drive_0184_sync',
                 '2011_09_28_drive_0185_sync',
                 '2011_09_28_drive_0186_sync',
                 '2011_09_28_drive_0187_sync',
                 '2011_09_28_drive_0191_sync',
                 '2011_09_28_drive_0192_sync',
                 '2011_09_28_drive_0195_sync',
                 '2011_09_28_drive_0198_sync',
                 '2011_09_28_drive_0199_sync',
                 '2011_09_28_drive_0201_sync',
                 '2011_09_28_drive_0204_sync',
                 '2011_09_28_drive_0205_sync',
                 '2011_09_28_drive_0208_sync',
                 '2011_09_28_drive_0209_sync',
                 '2011_09_28_drive_0214_sync',
                 '2011_09_28_drive_0216_sync',
                 '2011_09_28_drive_0220_sync',
                 '2011_09_28_drive_0222_sync']

    with open('kitti_continuous_train (1).txt') as oldfile,open('kitti_continuous_train (2).txt','w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)

    with open('kitti_continuous_test (1).txt') as oldfile,open('kitti_continuous_test (2).txt','w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)

    timer1 += time.time()

else:

    timer1 = -time.time()

    try:

        def read_text_file(filename,dataset_path):
            print("\n[Dataloader] Loading '%s'..." % filename)
            try:
                data = np.genfromtxt(filename,dtype='str',delimiter='\t')
                # print(data.shape)

                # Parsing Data
                image_filenames = list(data[:,0])
                depth_filenames = list(data[:,1])

                timer = -time.time()
                image_filenames = [dataset_path + filename for filename in image_filenames]
                depth_filenames = [dataset_path + filename for filename in depth_filenames]
                timer += time.time()
                print('time:',timer,'s\n')

            except OSError:
                raise OSError("Could not find the '%s' file." % filename)

            return image_filenames,depth_filenames


        image_filenames,depth_filenames = read_text_file(
            filename='/media/olorin/Documentos/raul/SIDE/kitti_continuous_train (2).txt',
            dataset_path='/media/olorin/Documentos/datasets/kitti/raw_data/')

        image_validation,depth_validation = read_text_file(
            filename='/media/olorin/Documentos/raul/SIDE/kitti_continuous_test (2).txt',
            dataset_path='/media/olorin/Documentos/datasets/kitti/raw_data/')

        image = sorted(image_filenames)
        depth = sorted(depth_filenames)
        image_val = sorted(image_validation)
        depth_val = sorted(depth_validation)

        train_images = image
        train_labels = depth
        test_images = image_val
        test_labels = depth_val

        print(len(image))
        print(len(depth))

        timer1 += time.time()

    except OSError:
        raise SystemExit

# imgs_A = load_and_scale_image(train_images[0])
# imgs_B = load_and_scale_depth(train_labels[0])

# imgs_A, imgs_B = np.array(imgs_A), np.array(imgs_B)

# print(imgs_A.shape)

# print(imgs_B.shape)

def sample_images():
    r,c = 2,3

    imgs_A = load_and_scale_image(train_images[0])
    imgs_B = load_and_scale_depth(train_labels[0])

    # Demo (for GIF)
    # imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
    # imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

    # Translate images to the other domain
    fake_B = g_AB.predict(imgs_A)
    fake_A = g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = g_BA.predict(fake_B)
    reconstr_B = g_AB.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A,reconstr_A,fake_A])

    gen_depths = np.concatenate([fake_B,imgs_B,reconstr_B])

    # Rescale images 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5
    # gen_depths = 0.5 * gen_depths + 0.5

    titles = ['Original','Translated','Reconstructed']
    fig,axs = plt.subplots(r,c)
    axs[0,0].imshow(gen_imgs[0])
    axs[0,0].set_title(titles[0])
    # axs[0,0].axis('off')
    axs[0,1].plot()
    cax0 = axs[0,1].imshow(np.squeeze(gen_depths[0], axis=2)*90.0)
    fig.colorbar(cax0,ax=axs[0,1])
    axs[0,1].set_title(titles[1])
    # axs[0,1].axis('off')
    axs[0,2].imshow(gen_imgs[1])
    axs[0,2].set_title(titles[2])
    # axs[0,2].axis('off')
    axs[1,0].plot()
    cax1 = axs[1,0].imshow(np.squeeze(gen_depths[1],axis=2)*90.0)
    fig.colorbar(cax1,ax=axs[1,0])
    axs[1,0].set_title(titles[0])
    # axs[1,0].axis('off')
    axs[1,1].imshow(gen_imgs[2])
    axs[1,1].set_title(titles[1])
    # axs[1,1].axis('off')
    axs[1,2].plot()
    cax2 = axs[1,2].imshow(np.squeeze(gen_depths[2],axis=2)*90.0)
    fig.colorbar(cax2,ax=axs[1,2])
    axs[1,2].set_title(titles[2])
    # axs[1,2].axis('off')
    # fig.savefig("images/%s/%d_%d.png" % (self.dataset_name,epoch,batch_i))
    plt.show()
    plt.pause(60.0)
    plt.close('all')

batch_start = 0
batch_end = batch_size

numSamples = len(train_images)

for epoch in range(epochs):
    for batch in tqdm(range((len(train_images)//batch_size)+1)):

        limit = min(batch_end,numSamples)

        imgs_A = np.concatenate(list(map(load_and_scale_image,train_images[batch_start:limit])),0)
        imgs_B = np.concatenate(list(map(load_and_scale_depth,train_labels[batch_start:limit])),0)

        # print(imgs_A.shape)
        # print(imgs_B.shape)

        batch_start += 1
        batch_end += 1

        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Translate images to opposite domain
        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = d_A.train_on_batch(imgs_A,valid)
        dA_loss_fake = d_A.train_on_batch(fake_A,fake)
        dA_loss = 0.5 * np.add(dA_loss_real,dA_loss_fake)

        dB_loss_real = d_B.train_on_batch(imgs_B,valid)
        dB_loss_fake = d_B.train_on_batch(fake_B,fake)
        dB_loss = 0.5 * np.add(dB_loss_real,dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * np.add(dA_loss,dB_loss)

        # ------------------
        #  Train Generators
        # ------------------

        # Train the generators
        g_loss = combined.train_on_batch([imgs_A,imgs_B],
                                              [valid,valid,
                                                imgs_A,imgs_B])

        elapsed_time = datetime.datetime.now() - start_time

    # Plot the progress
    print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f] time: %s " \
        % (epoch,epochs,
        d_loss[0],100 * d_loss[1],
        g_loss[0],
        np.mean(g_loss[1:3]),
        np.mean(g_loss[3:5]),
        elapsed_time))

        # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        g_AB.save_weights('g_AB_weights.h5')
        g_BA.save_weights('g_BA_weights.h5')
        d_A.save_weights('d_A_weights.h5')
        d_B.save_weights('d_B_weights.h5')
        combined.save_weights('combined_weights.h5')
        sample_images()
