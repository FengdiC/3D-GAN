import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tensorflow.contrib.slim as slim
from progress.bar import Bar
from PIL import Image
from keras.utils import np_utils
import time;
import math


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


class image_loader(object):
    def __init__(self, image_shape=(64, 64, 3), distortion_range=[0, 0.1], labels=None,
                 data_dir="../Data/faces_matrix.npy", raw_images_dir="../Data/faces_full/"):
        train_x = self.load_faces_raw(image_shape, data_dir, raw_images_dir);
        perm = np.random.permutation(train_x.shape[0]);
        train_x[:] = train_x[perm];
        if labels != None:
            self.train_y = self.load_labels(data_dir=labels);
            self.train_y[:] = self.train_y[perm];
        else:
            self.train_y = None;
        self.train_x = train_x;
        self.current_index = 0;
        self.image_shape = image_shape;
        self.data_dir = data_dir;
        self.num_data = train_x.shape[0];
        self.distortion_range = distortion_range;

    def load_labels(self, data_dir="../Data/painters_label.npy"):
        train_y = np.load(data_dir);
        # label_1 = np.ndarray.astype(train_y[:, 0], dtype=np.uint32);
        # label_1 = to_proper_labels(label_1, np.max(label_1) + 1);
        # label_2 = np.ndarray.astype(train_y[:, 1], dtype=np.uint32);
        # label_2 = to_proper_labels(label_2, np.max(label_2) + 1);
        # return np.concatenate((label_1, label_2), axis=1);
        label_1 = np.ndarray.astype(train_y[:], dtype=np.uint32);
        label_1 = to_proper_labels(label_1, np.max(label_1) + 1);
        return label_1;

    def next_batch(self, batch_size):
        with tf.device('/cpu:0'):
            batch = np.zeros((batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]),
                             dtype=np.float32);
            #labels = np.zeros((batch_size, self.train_y.shape[1]), dtype=np.float32);
            if self.current_index + batch_size < self.num_data:
                batch[:] = self.train_x[self.current_index:(self.current_index + batch_size)];
                #labels[:] = self.train_y[self.current_index:(self.current_index + batch_size)];
                self.current_index += batch_size;
            else:
                remain = self.num_data - self.current_index;
                start = batch_size - remain;
                batch[:remain] = self.train_x[self.current_index:self.num_data];
                batch[remain:] = self.train_x[0:start];
                # labels[:remain] = self.train_y[self.current_index:self.num_data];
                # labels[remain:] = self.train_y[0:start];
                perm = np.random.permutation(self.train_x.shape[0]);
                self.train_x[:] = self.train_x[perm];
                # self.train_y[:] = self.train_y[perm];
                self.current_index = start;
            return batch#, labels;

    def load_faces_raw(self, shape, directory, raw_images_dir):
        if os.path.isfile(directory) == True:
            result = np.load(directory);
            if result.shape[1] == shape[0] or result.shape[2] == shape[1]:
                return result;
        files = os.listdir(raw_images_dir);
        size = len(files);
        result = np.zeros((size, shape[0], shape[1], shape[2]));
        y_train = np.zeros((size, 1));
        index = 0;
        bar = Bar('Processing', max=len(files))
        for f in files:
            image = Image.open(raw_images_dir + f);
            image = image.resize((shape[0], shape[1]), Image.ANTIALIAS);
            image = np.array(image)
            result[index, :, :, :] = image[:, :, :];
            index += 1;
            bar.next();
        bar.finish();
        result = result.astype(np.float32);
        result = normalization(result);
        np.save(directory, result);
        return result;


def normalization(X):
    X = X / 255.
    X = (X - 0.5) / 0.5
    return X


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def deconv2d(input_, output_dim, kernel=(5, 5), strides=(2, 2), init=0.02, reuse=False, layer_name="deconvolution2d"):
    with tf.variable_scope(layer_name, reuse=reuse):
        w = tf.get_variable('w', [kernel[0], kernel[1], output_dim[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=init))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_dim, strides=[1, strides[0], strides[1], 1]);
        biases = tf.get_variable('biases', [output_dim[-1]], initializer=tf.truncated_normal_initializer(stddev=init));
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv;


def conv2d(input_, output_dim, kernel=(5, 5), strides=(2, 2), init=0.02, reuse=False, layer_name="convolution2d"):
    with tf.variable_scope(layer_name, reuse=reuse):
        w = tf.get_variable('w', [kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=init));
        conv = tf.nn.conv2d(input_, w, strides=[1, strides[0], strides[1], 1], padding='SAME');
        biases = tf.get_variable("biases", [output_dim], initializer=tf.truncated_normal_initializer(stddev=init));
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv;


def dense(input_, output_dim, init=0.02, layer_name="dense_layer", reuse=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(layer_name, reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_dim], tf.float32,
                                 tf.random_normal_initializer(stddev=init))
        bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(init))
        return tf.matmul(input_, matrix) + bias


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def combine_images(generated_images):
    num = generated_images.shape[0]
    height = int(math.ceil(math.sqrt(num)))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], height * shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / height)
        j = index % height
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]
    return image


def to_proper_labels(labels, _max):
    ret = np.zeros((labels.shape[0], _max));
    for i in range(labels.shape[0]):
        ret[i, labels[i]] = 1.0;
    return ret;


def generate_random_labels(batch_size, data_label, max_label, z_dim):
    labels = np.random.uniform(-1, 1, size=[batch_size, z_dim]);
    labels[:, 0:max_label] = data_label[:];
    # labels = np.random.normal(loc=mean, scale=std, size=(batch_size, 179));
    # rand_label_1 = np.random.randint(1, 43, size=(batch_size, 1));
    # rand_label_1 = to_proper_labels(rand_label_1, 43);
    # rand_label_2 = np.random.randint(0, 179 - 43, size=(batch_size, 1));
    # rand_label_2 = to_proper_labels(rand_label_2, 179 - 43);
    # labels = np.concatenate((rand_label_1, rand_label_2), axis=1);
    # labels = np.clip(labels, 0, 1);
    return labels;


def sample_Z(m, n, opt_label=None, add_noise=False):
    if opt_label is None:
        return np.random.uniform(-1., 1., size=[m, n])
    else:
        input = np.random.uniform(-1, 1, size=(m, n));
        input[:, 0:(opt_label.shape[1])] = opt_label[:, :];
        return input;


def deconv_bn_lrelu(_input, output_shape, kernel=(5, 5), strides=(2, 2), bn_n="default_deconv", lname="default",
                    reuse=False):
    with tf.variable_scope("deconv_bn_lrelu"):
        l = deconv2d(_input, output_shape, kernel=kernel, strides=strides, layer_name=lname, reuse=reuse)
        # bn = batch_norm(name=bn_n);
        # l = lrelu(bn(l))
        l = lrelu(l);
        return l;


def conv_bn_lrelu(_input, filter_num, kernel=(5, 5), strides=(2, 2), bn_n="default_conv", lname="default", reuse=False):
    with tf.variable_scope("conv_bn_lrelu"):
        l = conv2d(_input, filter_num, kernel=kernel, strides=strides, layer_name=lname, reuse=reuse)
        # bn = batch_norm(name=bn_n);
        # l = lrelu(bn(l))
        l = lrelu(l);
        return l;


def save(saver, sess, checkpoint_dir, step):
    model_name = "custom.model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,
               os.path.join(checkpoint_dir, model_name),
               global_step=step)


def load(saver, sess, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    saver.restore(sess, checkpoint_dir);
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #     saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    #     print(" [*] Success to read {}".format(ckpt_name))
    #     return True
    # else:
    #     print(" [*] Failed to find a checkpoint")
    #     return False

def get_triple_img(full_image_matrix):
    img_32 = tf.layers.average_pooling2d(full_image_matrix, [2, 2], [2, 2]);
    img_16 = tf.layers.average_pooling2d(img_32, [2, 2], [2, 2]);
    return img_16, img_32, full_image_matrix;

def gan_generator_uncertainty(noise_input, batch_size, filter_num=64, reuse=False):
    with tf.variable_scope("generator") as scope:
        s_h1, s_w1 = 64, 64
        s_h2, s_w2 = 32, 32
        s_h3, s_w3 = 16, 16
        s_h4, s_w4 = 8, 8
        s_h5, s_w5 = 4, 4
        # project `z` and reshape
        print("Generator Layers: ");
        print(noise_input.shape);
        l = dense(noise_input, filter_num * s_h5 * s_w5, layer_name='dense_1', reuse=reuse)
        l = tf.reshape(l, [batch_size, s_h5, s_w5, filter_num])
        l = lrelu(l)
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h5, s_w5, filter_num * 16), kernel=(3, 3), strides=(1, 1), bn_n="bn_1", lname="conv1", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h4, s_w4, filter_num * 8), kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h3, s_w3, filter_num * 4), kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h2, s_w2, filter_num * 2), kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h1, s_w1, filter_num * 1), kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);
        print(l.shape);
        img_64 = tf.nn.tanh(conv2d(l, 4, kernel=(1, 1), strides=(1, 1), layer_name="img_64", reuse=reuse));
        image = img_64[:, :, :, :3];
        uncertainty = img_64[:, :, :, 3];
        return image, tf.nn.sigmoid(uncertainty);

def gan_generator(noise_input, batch_size, final_channel_num=3, filter_num=64, reuse=False):
    with tf.variable_scope("generator") as scope:
        s_h1, s_w1 = 64, 64
        s_h2, s_w2 = 32, 32
        s_h3, s_w3 = 16, 16
        s_h4, s_w4 = 8, 8
        s_h5, s_w5 = 4, 4
        # project `z` and reshape
        print("Generator Layers: ");
        print(noise_input.shape);
        l = dense(noise_input, filter_num * s_h5 * s_w5, layer_name='dense_1', reuse=reuse)
        l = tf.reshape(l, [batch_size, s_h5, s_w5, filter_num])
        l = lrelu(l)
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h5, s_w5, filter_num * 16), kernel=(3, 3), strides=(1, 1), bn_n="bn_1", lname="conv1", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h4, s_w4, filter_num * 8), kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h3, s_w3, filter_num * 4), kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h2, s_w2, filter_num * 2), kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h1, s_w1, filter_num * 1), kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);
        print(l.shape);
        l = tf.nn.tanh(conv2d(l, final_channel_num, kernel=(1, 1), strides=(1, 1), layer_name="img_64", reuse=reuse));
        print(l.shape);
        return l;


def gan_disc_16(image, batch_size, output_size=1, filter_num=64, reuse=False):
    with tf.variable_scope("discriminator_16") as scope:
        print("Discriminator 16 Layers: ");
        print(image.shape);
        l = conv_bn_lrelu(image, filter_num, kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);  # out 32, 32
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 2, kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);  # out 16
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 4, kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);  # out 8
        print(l.shape);
        l = tf.reshape(l, [batch_size, -1])
        l = dense(l, output_size, layer_name='dense_1', reuse=reuse);
        print(l.shape);
        return l;


def gan_disc_32(image, batch_size, output_size=1, filter_num=64, reuse=False):
    with tf.variable_scope("discriminator_32") as scope:
        print("Discriminator 32 Layers: ");
        print(image.shape);
        l = conv_bn_lrelu(image, filter_num, kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);  # out 32, 32
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 2, kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);  # out 16
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 4, kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);  # out 8
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 8, kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);  # out 4
        print(l.shape);
        l = tf.reshape(l, [batch_size, -1])
        l = dense(l, output_size, layer_name='dense_1', reuse=reuse);
        print(l.shape);
        return l;

def gan_disc_64(image, batch_size, output_size=1, filter_num=64, reuse=False):
    with tf.variable_scope("discriminator_64") as scope:
        print("Discriminator 64 Layers: ");
        print(image.shape);
        l = conv_bn_lrelu(image, filter_num, kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);  # out 32, 32
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 2, kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);  # out 16
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 4, kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);  # out 8
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 6, kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);  # out 4
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 8, kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);  # out 2
        l = tf.reshape(l, [batch_size, -1])
        l = dense(l, output_size, layer_name='dense_1', reuse=reuse);

        print(l.shape);
        return l;

def gan_disc_uncertainty_64(image, batch_size, output_size=1, filter_num=64, reuse=False):
    with tf.variable_scope("discriminator_64") as scope:
        print("Discriminator 64 Layers: ");
        print(image.shape);
        l = conv_bn_lrelu(image, filter_num, kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);  # out 32, 32
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 2, kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);  # out 16
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 4, kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);  # out 8
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 6, kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);  # out 4
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 8, kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);  # out 2

        s_h1, s_w1 = 64, 64
        s_h2, s_w2 = 32, 32
        s_h3, s_w3 = 16, 16
        s_h4, s_w4 = 8, 8
        s_h5, s_w5 = 4, 4
        s_h6, s_w6 = 2, 2
        l = deconv_bn_lrelu(l, (batch_size, s_h6, s_w6, filter_num * 8), kernel=(3, 3), strides=(1, 1), bn_n="bn_0", lname="conv0", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h5, s_w5, filter_num * 8), kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h4, s_w4, filter_num * 6), kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h3, s_w3, filter_num * 4), kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h2, s_w2, filter_num * 2), kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h1, s_w1, filter_num * 1), kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);
        print(l.shape);
        l = tf.nn.tanh(conv2d(l, 1, kernel=(1, 1), strides=(1, 1), layer_name="img_64", reuse=reuse));
        print(l.shape);

        l = tf.reshape(l, [batch_size, -1]);
        print(l.shape);
        l = dense(l, s_h1 * s_w1, layer_name='dense_1', reuse=reuse);
        l = tf.reshape(l, [batch_size, s_h1, s_w1, 1]);
        print(l.shape);
        return l;


def build_autoencoder_gan_discriminator_32(image, batch_size, output_size=1, filter_num=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        print("Discriminator Layers: ");
        print(image.shape);
        l = conv_bn_lrelu(image, filter_num, kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);  # out 32, 32
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 2, kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);  # out 16
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 4, kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);  # out 8
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 8, kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);  # out 4
        print(l.shape);
        l = tf.reshape(l, [batch_size, -1])
        l = dense(l, output_size, layer_name='dense_1', reuse=reuse);
        print(l.shape);
        wass_logits, label_logits = tf.split(l, [1, output_size - 1], 1);
        return wass_logits, label_logits;


def build_autoencoder_gan_generator_32(noise_input, batch_size, filter_num=64, reuse=False):
    with tf.variable_scope("generator") as scope:
        s_h2, s_w2 = 32, 32
        s_h3, s_w3 = 16, 16
        s_h4, s_w4 = 8, 8
        s_h5, s_w5 = 4, 4
        # project `z` and reshape
        print("Generator Layers: ");
        print(noise_input.shape);
        l = dense(noise_input, filter_num * s_h5 * s_w5 * 2, layer_name='dense_1', reuse=reuse)
        l = tf.reshape(l, [batch_size, s_h5, s_w5, filter_num * 2])
        l = lrelu(l)
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h5, s_w5, filter_num * 16), kernel=(3, 3), strides=(1, 1), bn_n="bn_1",
                            lname="conv1", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h4, s_w4, filter_num * 8), kernel=(3, 3), bn_n="bn_2", lname="conv2",
                            reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h3, s_w3, filter_num * 4), kernel=(3, 3), bn_n="bn_3", lname="conv3",
                            reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h2, s_w2, filter_num * 2), kernel=(3, 3), bn_n="bn_4", lname="conv4",
                            reuse=reuse);
        print(l.shape);
        l = conv2d(l, 3, kernel=(1, 1), strides=(1, 1), layer_name="final_conv", reuse=reuse)
        print(l.shape);
        # l = deconv2d(l, (batch_size, s_h1, s_w1, 3), strides=(1, 1), kernel=(3, 3), layer_name="output_conv", reuse=False)
        # print(l.shape);
        return tf.nn.tanh(l)



def build_autoencoder_gan_generator(noise_input, batch_size, final_channel_num=3, filter_num=64, reuse=False):
    with tf.variable_scope("generator") as scope:
        s_h1, s_w1 = 64, 64
        s_h2, s_w2 = 32, 32
        s_h3, s_w3 = 16, 16
        s_h4, s_w4 = 8, 8
        s_h5, s_w5 = 4, 4
        # project `z` and reshape
        print("Generator Layers: ");
        print(noise_input.shape);
        l = dense(noise_input, filter_num * s_h5 * s_w5 * 2, layer_name='dense_1', reuse=reuse)
        l = tf.reshape(l, [batch_size, s_h5, s_w5, filter_num * 2])
        l = lrelu(l)
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h5, s_w5, filter_num * 16), kernel=(3, 3), strides=(1, 1), bn_n="bn_1",
                            lname="conv1", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h4, s_w4, filter_num * 8), kernel=(3, 3), bn_n="bn_2", lname="conv2",
                            reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h3, s_w3, filter_num * 4), kernel=(3, 3), bn_n="bn_3", lname="conv3",
                            reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h2, s_w2, filter_num * 2), kernel=(3, 3), bn_n="bn_4", lname="conv4",
                            reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h1, s_w1, filter_num * 1), kernel=(3, 3), bn_n="bn_5", lname="conv5",
                            reuse=reuse);
        print(l.shape);
        l = conv2d(l, 3, kernel=(1, 1), strides=(1, 1), layer_name="final_conv", reuse=reuse)
        print(l.shape);
        # l = deconv2d(l, (batch_size, s_h1, s_w1, 3), strides=(1, 1), kernel=(3, 3), layer_name="output_conv", reuse=False)
        # print(l.shape);
        return tf.nn.tanh(l)


def build_autoencoder_gan_discriminator(image, batch_size, output_size=1, filter_num=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        print("Discriminator Layers: ");
        print(image.shape);
        l = conv_bn_lrelu(image, filter_num, kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);  # out 32, 32
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 2, kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);  # out 16
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 4, kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);  # out 8
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 8, kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);  # out 4
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 16, kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);  # out 2
        print(l.shape);
        # l = conv_bn_lrelu(l, filter_num*16, kernel=(3, 3), strides=(2, 2), bn_n="bn_6", lname="conv6", reuse=reuse); #out 2
        # print(l.shape);
        l = tf.reshape(l, [batch_size, -1])
        l = dense(l, output_size, layer_name='dense_1', reuse=reuse);
        print(l.shape);
        return l;


def encoding_loss(original_encoding, resulting_encoding):
    return tf.reduce_mean(np.abs(original_encoding - resulting_encoding));


def pixel_wise_loss(image, decoded_image):
    return tf.reduce_mean(np.abs(image - decoded_image));


def hinge_loss(disc_real_logit, disc_fake_logit, Y):
    D_loss = tf.reduce_mean(tf.maximum(0.0, Y - disc_real_logit)) + tf.reduce_mean(
        tf.maximum(0.0, Y + disc_fake_logit));
    G_loss = tf.reduce_mean(tf.maximum(0.0, Y - disc_fake_logit))
    return D_loss, G_loss;


def softmax_loss(disc_real_logit, disc_fake_logit, Y):
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=disc_real_logit))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=disc_fake_logit))
    return D_loss, G_loss


def wasserstein_16(real_data, fake_data, disc_fake, disc_real, batch_size, output_size, _lambda=5, label_cost_ratio=20, _gamma=0.01):
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[batch_size, 16, 16, 3],
        minval=0.1,
        maxval=1.0
    )
    differences = fake_data - real_data
    print(differences.shape);
    interpolates = real_data + (alpha * differences)

    sampler = gan_disc_16(interpolates, batch_size, output_size=output_size, reuse=True);
    gradients = tf.gradients(sampler, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    extra_cost = _lambda * (gradient_penalty);

    disc_cost += extra_cost
    return disc_cost, gen_cost;

def wasserstein_32(real_data, fake_data, disc_fake, disc_real, batch_size, output_size, _lambda=5, label_cost_ratio=20, _gamma=0.01):
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[batch_size, 32, 32, 3],
        minval=0.1,
        maxval=1.0
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)

    sampler = gan_disc_32(interpolates, batch_size, output_size=output_size, reuse=True);
    gradients = tf.gradients(sampler, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    extra_cost = _lambda * (gradient_penalty);

    disc_cost += extra_cost
    return disc_cost, gen_cost;

def wasserstein_64(real_data, fake_data, disc_fake, disc_real, batch_size, output_size, _lambda=5, label_cost_ratio=20, _gamma=0.01):
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[batch_size, 64, 64, 3],
        minval=0.1,
        maxval=1.0
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)

    sampler = gan_disc_64(interpolates, batch_size, output_size=output_size, reuse=True);
    gradients = tf.gradients(sampler, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    extra_cost = _lambda * (gradient_penalty);

    disc_cost += extra_cost
    return disc_cost, gen_cost;

def wasserstein_uncertainty_64(real_data, fake_data, disc_fake, disc_real, reg_loss, batch_size, output_size, _lambda=5, label_cost_ratio=20, _gamma=0.01):
    gen_cost = -tf.reduce_mean(disc_fake) + reg_loss;
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[batch_size, 64, 64, 3],
        minval=0.1,
        maxval=1.0
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)

    sampler = gan_disc_uncertainty_64(interpolates, batch_size, output_size=output_size, reuse=True);
    gradients = tf.gradients(sampler, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    extra_cost = _lambda * (gradient_penalty);

    # disc_cost += extra_cost
    return disc_cost, gen_cost;



def train(num_epoch=500, learning_rate=0.0002, _lambda=10, image_dim=(64, 64, 3), batch_size=64, output_size=1,
          Z_dim=(512)):
    Z = tf.placeholder(tf.float32, shape=[batch_size, Z_dim])
    X = tf.placeholder(tf.float32, shape=[batch_size, image_dim[0], image_dim[1], image_dim[2]])

    G = gan_generator(Z, batch_size, final_channel_num=image_dim[2]);
    d_logit_real_64 = gan_disc_64(X, batch_size, output_size=1);
    d_logit_fake_64 = gan_disc_64(G, batch_size, output_size=1, reuse=True);

    # uncertainty = tf.expand_dims(uncertainty, axis=3);
    # disc_uncertainty_image_fake *= uncertainty;
    # Regularization Loss:
    # reg_loss = tf.reduce_sum(1 - uncertainty)
    
    D_loss = 0.5 * tf.reduce_mean((d_logit_real_64 - 1) ** 2) + tf.reduce_mean((d_logit_fake_64 ** 2) * uncertainty);
    G_loss = 0.5 * tf.reduce_mean(((d_logit_fake_64 - 1) ** 2) * uncertainty) + reg_loss;

    #D_loss_64, G_loss_64 = wasserstein_uncertainty_64(X, G, disc_uncertainty_image_fake, disc_uncertainty_image_real, reg_loss, batch_size, output_size=output_size);


    #D_loss_16, G_loss_16 = wasserstein_16(real_img_16, img_16, d_logit_fake_16, d_logit_real_16, batch_size, output_size=output_size);
    #D_loss_32, G_loss_32 = wasserstein_32(real_img_32, img_32, d_logit_fake_32, d_logit_real_32, batch_size, output_size=output_size);
    # D_loss, G_loss = wasserstein_64(X, G, d_logit_fake_64, d_logit_real_64, batch_size, output_size=output_size);
    # D_loss = D_loss_64;
    # G_loss = G_loss_64 + tf.reduce_mean(tf.abs(uncertainty));

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]

    D_solver = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=g_vars)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    fixed_sample = sample_Z(batch_size, Z_dim);
    samples = sess.run(G, feed_dict={Z: fixed_sample});
    image = combine_images(samples)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save('../Figures/epoch_' + str(-1) + ".png")

    data = image_loader(data_dir="../Data/edges_matrix.npy", image_shape=image_dim);
    num_batches = data.num_data // batch_size;
    num_gen = 1;
    num_disc = 5;
    batch_count = 0;
    for epoch in range(num_epoch):
        start_time = time.time();
        total_d_loss = [];
        total_g_loss = [];
        reg_val = [];
        bar = Bar('Processing', max=num_batches)
        for batch in range(num_batches):
            for i in range(num_disc):
                image_batch = data.next_batch(batch_size);
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: image_batch, Z: sample_Z(batch_size, Z_dim)})
                total_d_loss.append(D_loss_curr);
            for i in range(num_gen):
                _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})
                total_g_loss.append(G_loss_curr);
            batch_count += 1;
            if batch_count % 50 == 0:
                samples = sess.run(G, feed_dict={Z: fixed_sample});
                image = combine_images(samples)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save('../Figures/epoch_' + str(epoch) + ".png")
                print("\n")
                print('D loss: ', np.mean(total_d_loss))
                print('G_loss: ', np.mean(total_g_loss))
                print("Convergence: ", abs(np.mean(total_d_loss) - np.mean(total_g_loss)));
            bar.next();
        bar.finish();
        print("Num Generation: ", num_gen);
        print('Iter: {}'.format(epoch))
        print("Time Elapsed: ", time.time() - start_time)
        print('D loss: ', np.mean(total_d_loss))
        print('G_loss: ', np.mean(total_g_loss))
        save(saver, sess, "../Models", epoch)
        print()









    # G_sample = build_autoencoder_gan_generator_32(Z, batch_size)
    # D_logit_real, label_logit_real = build_autoencoder_gan_discriminator_32(X, batch_size, output_size=output_size + 1)
    # D_logit_fake, label_logit_fake = build_autoencoder_gan_discriminator_32(G_sample, batch_size,
    #                                                                         output_size=output_size + 1, reuse=True)
    # D_loss, G_loss, d_label_loss, g_label_loss = wasserstein(X, G_sample, D_logit_real, D_logit_fake, Y,
    #                                                          label_logit_real, label_logit_fake, batch_size,
    #                                                          output_size + 1, _lambda=_lambda);

    # t_vars = tf.trainable_variables()
    # d_vars = [var for var in t_vars if 'discriminator' in var.name]
    # g_vars = [var for var in t_vars if 'generator' in var.name]

    # D_solver = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=d_vars)
    # G_solver = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=g_vars)

    # saver = tf.train.Saver()
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # # load(saver, sess, "../Models/custom.model-131");
    # labels = np.zeros((batch_size, 1), dtype=np.uint16);
    # labels = np.squeeze(labels);
    # for i in range(batch_size):
    #     labels[i] = 4;
    # labels = to_proper_labels(labels, 10);
    # fixed_sample = generate_random_labels(batch_size, labels, 10, Z_dim)
    # samples = sess.run(G_sample, feed_dict={Z: fixed_sample});
    # image = combine_images(samples)
    # image = image * 127.5 + 127.5
    # Image.fromarray(image.astype(np.uint8)).save('../Figures/epoch_' + str(1) + ".png")

    # num_batches = data.num_data // batch_size;
    # num_gen = 1;
    # num_disc = 5;
    # batch_count = 0;
    # for epoch in range(num_epoch):
    #     start_time = time.time();
    #     total_d_loss = [];
    #     total_g_loss = [];
    #     bar = Bar('Processing', max=num_batches)
    #     for batch in range(num_batches):
    #         for i in range(num_disc):
    #             image_batch, labels = data.next_batch(batch_size);
    #             generated_z = generate_random_labels(batch_size, labels, 10, Z_dim)
    #             _, D_loss_curr, g_l_loss, d_l_loss = sess.run([D_solver, D_loss, d_label_loss, g_label_loss],
    #                                                           feed_dict={Y: labels, X: image_batch, Z: generated_z})
    #             total_d_loss.append(D_loss_curr);
    #         for i in range(num_gen):
    #             generated_z = generate_random_labels(batch_size, labels, 10, Z_dim)
    #             _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Y: labels, Z: generated_z})
    #             total_g_loss.append(G_loss_curr);
    #         batch_count += 1;
    #         if batch_count % 50 == 0:
    #             samples = sess.run(G_sample, feed_dict={Z: fixed_sample});
    #             image = combine_images(samples)
    #             image = image * 127.5 + 127.5
    #             Image.fromarray(image.astype(np.uint8)).save('../Figures/epoch_' + str(epoch) + ".png")
    #             print('\nG Label Cost: ', g_l_loss);
    #             print('D Label Cost: ', d_l_loss);
    #             print('D loss: ', np.mean(total_d_loss))
    #             print('G_loss: ', np.mean(total_g_loss))
    #             print("Convergence: ", abs(np.mean(total_d_loss) - np.mean(total_g_loss)));
    #         bar.next();
    #     bar.finish();
    #     print("Num Generation: ", num_gen);
    #     print('Iter: {}'.format(epoch))
    #     print("Time Elapsed: ", time.time() - start_time)
    #     print('D loss: ', np.mean(total_d_loss))
    #     print('G_loss: ', np.mean(total_g_loss))
    #     save(saver, sess, "../Models", epoch)
    #     print()


train();
