import tensorflow as tf
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import time
import os
import tensorflow.contrib.slim as slim
import hdf5storage
from progress.bar import Bar
from keras.layers import Concatenate


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
    # assume RGBDE has hsape (N*W*H*5)
    def __init__(self, image_shape=(240, 320, 5), distortion_range=[0, 0.1],
                 data_dir="../../Data/RGBDE.npy"):
        images = np.load(data_dir)
        perm = np.random.permutation(images.shape[3])
        images[:,:,:,:] = images[:,:,:,perm]
        train_x = np.ones((image_shape[0],image_shape[1],image_shape[2],images.shape[3]))
        #label = np.load(label_dir)
        #label[:,:,:] = label[:,:,perm]
        #train_y = np.ones((label.shape[2], image_shape[0],image_shape[1],1))
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i,:,:,:])
            img = img.resize((image_shape[1], image_shape[2]), 5);
            img = np.array(img)
            train_x[i,:,:,:] = img[:,:,:]

            #lb = Image.fromarray(label[:, :, i]*255/np.max(label[:, :, i]))
            #lb = lb.resize((image_shape[1], image_shape[0]))
            #lb = np.array(lb)
            #lb = np.expand_dims(lb,axis=-1)
            #train_y[i,:,:,:] = np.where(lb>0,np.ones((lb.shape)),np.zeros((lb.shape)))
        #self.train_y = train_y
        train_x = normalization(train_x)
        self.train_x = train_x
        self.current_index = 0
        self.image_shape = image_shape
        self.data_dir = data_dir
        self.num_data = train_x.shape[0]
        self.distortion_range = distortion_range;


    def next_batch(self, batch_size):
        with tf.device('/cpu:0'):
            batch = np.zeros((batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]),
                             dtype=np.float32);
            #labels = np.zeros((batch_size, self.train_y.shape[1],self.train_y.shape[2],1), dtype=np.float32);
            if self.current_index + batch_size < self.num_data:
                batch[:] = self.train_x[self.current_index:(self.current_index + batch_size)];
                #labels[:] = self.train_y[self.current_index:(self.current_index + batch_size)];
                self.current_index += batch_size;
            else:
                remain = self.num_data - self.current_index;
                start = batch_size - remain;
                batch[:remain] = self.train_x[self.current_index:self.num_data];
                batch[remain:] = self.train_x[0:start];
                #labels[:remain] = self.train_y[self.current_index:self.num_data];
                #labels[remain:] = self.train_y[0:start];
                perm = np.random.permutation(self.train_x.shape[0]);
                self.train_x[:] = self.train_x[perm];
                #self.train_y[:] = self.train_y[perm];
                self.current_index = start;
            return batch#, labels


def normalization(X):
    X[:,:,:,0:3] = X[:,:,:,0:3] / 255.
    X[:,:,:,0:3] = (X[:,:,:,0:3] - 0.5) / 0.5
    X[:,:,:,4] = np.divide(X[:,:,:,4],np.max(X[:,:,:,4],axis=0))
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


def conv2d(input_, output_dim, kernel=(3, 3), strides=(2, 2), init=0.02, reuse=False, layer_name="convolution2d"):
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


def deconv_bn_lrelu(_input, output_shape, kernel=(3, 3), strides=(2, 2), bn_n="default_deconv", lname="default",
                    reuse=False):
    with tf.variable_scope("deconv_bn_lrelu"):
        l = deconv2d(_input, output_shape, kernel=kernel, strides=strides, layer_name=lname, reuse=reuse)
        # bn = batch_norm(name=bn_n);
        # l = lrelu(bn(l))
        l = lrelu(l);
        return l;


def conv_bn_lrelu(_input, filter_num, kernel=(3, 3), strides=(2, 2), bn_n="default_conv", lname="default", reuse=False):
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

def gan_generator(noise_input, batch_size, final_channel_num=5, filter_num=64, reuse=False):
    with tf.variable_scope("generator") as scope:
        s_h1, s_w1 = 256, 320
        s_h2, s_w2 = 128, 160
        s_h3, s_w3 = 64, 80
        s_h4, s_w4 = 32, 40
        s_h5, s_w5 = 16, 20
        s_h6, s_w6 = 8,10
        s_h7, s_w7 = 4, 5

        # project `z` and reshape
        print("Generator Layers: ");
        print(noise_input.shape);
        l = dense(noise_input, filter_num * s_h7 * s_w7, layer_name='dense_1', reuse=reuse)
        l = tf.reshape(l, [batch_size, s_h7, s_w7, filter_num])
        l = lrelu(l)
        l = deconv_bn_lrelu(l, (batch_size, s_h7, s_w7, filter_num * 16), kernel=(3, 3), strides=(1, 1), bn_n="bn_1",lname="conv1", reuse=reuse)
        l = deconv_bn_lrelu(l, (batch_size, s_h6, s_w6, filter_num * 8), kernel=(3, 3), bn_n="bn_2", lname="conv2",reuse=reuse)
        l = deconv_bn_lrelu(l, (batch_size, s_h5, s_w5, filter_num * 16), kernel=(3, 3), strides=(1, 1), bn_n="bn_3", lname="conv3", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h4, s_w4, filter_num * 8), kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h3, s_w3, filter_num * 4), kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h2, s_w2, filter_num * 2), kernel=(3, 3), bn_n="bn_6", lname="conv6", reuse=reuse);
        print(l.shape);
        l = deconv_bn_lrelu(l, (batch_size, s_h1, s_w1, filter_num * 1), kernel=(3, 3), bn_n="bn_7", lname="conv7", reuse=reuse);
        print(l.shape);
        l = tf.nn.tanh(conv2d(l, final_channel_num, kernel=(1, 1), strides=(1, 1), layer_name="img_64", reuse=reuse));
        print(l.shape);
        return l

def gan_disc_256(image, batch_size, output_size=1, filter_num=64, reuse=False):
    with tf.variable_scope("discriminator_256") as scope:
        print("Discriminator 256 Layers: ");
        print(image.shape);
        l = conv_bn_lrelu(image, filter_num, kernel=(3, 3), bn_n="bn_1", lname="conv1", reuse=reuse);  # out 128
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 2, kernel=(3, 3), bn_n="bn_2", lname="conv2", reuse=reuse);  # out 64
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 4, kernel=(3, 3), bn_n="bn_3", lname="conv3", reuse=reuse);  # out 32
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 6, kernel=(3, 3), bn_n="bn_4", lname="conv4", reuse=reuse);  # out 16
        print(l.shape);
        l = conv_bn_lrelu(l, filter_num * 8, kernel=(3, 3), bn_n="bn_5", lname="conv5", reuse=reuse);  # out 8
        l = conv_bn_lrelu(l, filter_num * 16, kernel=(3, 3), bn_n="bn_6", lname="conv6", reuse=reuse);  # out 4
        l = conv_bn_lrelu(l, filter_num * 32, kernel=(3, 3), bn_n="bn_7", lname="conv7", reuse=reuse);  # out 2
        l = tf.reshape(l, [batch_size, -1])
        l = dense(l, output_size, layer_name='dense_1', reuse=reuse);

        print(l.shape);
        return l

def wasserstein_256(real_data, fake_data, disc_fake, disc_real, weight , batch_size,output_size, _lambda=10,  _gamma=0.01):

    #regularization of gen_cost
    K = np.array([[1.7604, 0, 0, 0], [0, 1.7604, 0, 0], [0, 0, 1, 0]])
    (n,m) = (fake_data[1],fake_data[2])
    E = fake_data[:,:,:,4]
    D = fake_data[:, :, :, 3]
    I = fake_data[:, :, :, 0:3]
    Ldx,Ldy,Lix,Liy,Le =0
    for i in range(n-1)+1:
        for j in range(m-1)+1:
            kx,ky = np.ones(E.shape[0]) - E[:,i,j]
            count = 0
            ldx, ldy, lix, liy = 0
            for num in range(4)+1:
                if(i-num<0 or i+num>n-1 or j-num<0 or j-num >m-1):
                    continue
                count +=1
                #edge function
                kx_curr = np.ones(E.shape[0])- np.maximum(np.maximum(E[:,i-num,j],kx),np.maximum(E[:,i+num,j],kx))
                kx = kx_curr
                ky_curr = np.ones(E.shape[0]) - np.maximum(np.maximum(E[:, i, j-num], ky),
                                                           np.maximum(E[:, i, j+num], ky))
                ky = ky_curr

                #second order derivatice of depth
                left = D[:,i-num,j] *np.linalg.inv(K).dot(np.array([i-num,j,1]))
                right = D[:, i + num, j] * np.linalg.inv(K).dot(np.array([i + num, j, 1]))
                middle = D[:,i,j] *np.linalg.inv(K).dot(np.array([i,j,1]))
                up = D[:, i, j+num] * np.linalg.inv(K).dot(np.array([i, j+num, 1]))
                down = D[:, i, j - num] * np.linalg.inv(K).dot(np.array([i, j - num, 1]))

                ldx += np.abs((D[:,i+num,j]-D[:,i,j])/np.dot(right-middle,right-middle)-
                              (D[:,i,j]-D[:,i-num,j])/np.dot(middle-left,middle-left)) * kx
                ldy += np.abs((D[:, i, j+num] - D[:, i, j]) / np.dot(up - middle, up - middle) -
                              (D[:, i, j] - D[:, i , j-num]) / np.dot(middle - down, middle - down))*ky
                lix += (np.linalg.norm((I[:,i+num,j]-I[:,i,j])/np.sqrt(np.dot(right-middle,right-middle)),ord=1)+
                    np.linalg.norm((I[:,i,j]-I[:,i-num,j])/np.sqrt(np.dot(middle-left,middle-left)),ord=1) )  * kx
                liy += (np.linalg.norm((I[:, i , j+num] - I[:, i, j]) / np.sqrt(np.dot(up - middle, up - middle)), ord=1) +
                    np.linalg.norm((I[:, i, j] - I[:, i , j-num]) / np.sqrt(np.dot(middle - down, middle - down)),ord=1)) * ky
            Ldx += ldx/count
            Ldy += ldy/count
            Lix += lix /(2*count)
            Liy += liy/(2*count)
            Le += np.dot(E[:,i,j],E[:,i,j])

    loss = -disc_fake + weight*(Ldx +Ldy + Lix + Liy + Le)
    gen_cost = -tf.reduce_mean(loss)

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    alpha = tf.random_uniform(
        shape=[batch_size, 256, 256, 3],
        minval=0.1,
        maxval=1.0
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)

    sampler = gan_disc_256(interpolates, batch_size, output_size=output_size, reuse=True);
    gradients = tf.gradients(sampler, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    extra_cost = _lambda * (gradient_penalty);

    disc_cost += extra_cost
    return disc_cost, gen_cost, [Ldx,Ldy,Lix,Liy,Le]


def train(num_epoch=500, learning_rate=0.0002, _lambda=10, image_dim=(256, 320, 5), batch_size=16, output_size=1,
          Z_dim=(512)):
    Z = tf.placeholder(tf.float32, shape=[batch_size, Z_dim])
    X = tf.placeholder(tf.float32, shape=[batch_size, image_dim[0], image_dim[1], image_dim[2]])
    weight  = tf.placeholder(tf.float32, shape=[1,])

    G = gan_generator(Z, batch_size, final_channel_num=image_dim[2]);
    d_logit_real_256 = gan_disc_256(X, batch_size, output_size=1);
    d_logit_fake_256 = gan_disc_256(G, batch_size, output_size=1, reuse=True);

    D_loss, G_loss,l = wasserstein_256(X, G, d_logit_fake_256, d_logit_real_256,weight ,batch_size,output_size=output_size);

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
    Image.fromarray(image.astype(np.uint8)).save('../../Figures/epoch_' + str(-1) + ".png")

    data = image_loader(image_shape=image_dim)
    w = tf.zeros([1,])
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
                _, D_loss_curr = sess.run([D_solver, D_loss],
                                          feed_dict={X: image_batch, Z: sample_Z(batch_size, Z_dim), weight: w})
                total_d_loss.append(D_loss_curr);
            for i in range(num_gen):
                _, G_loss_curr,l_loss_curr = sess.run([G_solver, G_loss,l], feed_dict={Z: sample_Z(batch_size, Z_dim), weight: w})
                total_g_loss.append(G_loss_curr);
                reg_val.append(l_loss_curr)
            batch_count += 1;
            if batch_count % 50 == 0:

                w = tf.add_n(w,tf.constant([0.05]))
                samples = sess.run(G, feed_dict={Z: fixed_sample});
                image = combine_images(samples)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save('../../Figures/epoch_' + str(epoch) + ".png")
                print("\n")
                print('D loss: ', np.mean(total_d_loss))
                print('G_loss: ', np.mean(total_g_loss))

                print('reg_loss: ', map(mean, zip(*reg_val)))
                print("Convergence: ", abs(np.mean(total_d_loss) - np.mean(total_g_loss)));
            bar.next();
        bar.finish();
        print("Num Generation: ", num_gen);
        print('Iter: {}'.format(epoch))
        print("Time Elapsed: ", time.time() - start_time)
        print('D loss: ', np.mean(total_d_loss))
        print('G_loss: ', np.mean(total_g_loss))
        print('reg_loss: ', map(mean, zip(*reg_val)))
        save(saver, sess, "../Models", epoch)
        print()

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

def mean(a):
    a = sum(a) / len(a)
    return a