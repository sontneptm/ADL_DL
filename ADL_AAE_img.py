import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import pandas as pd
import sklearn
import numpy as np
import random
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import entropy
from scipy import signal
import ADL_PF as prior
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

# Gaussian MLP as encoder
def MLP_encoder(inputs, n_output, keep_prob):
    with tf.variable_scope("MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

         # 1st hidden layer
        ws = tf.get_variable('ws', [inputs.get_shape()[1],1024], initializer=w_init)
        bs = tf.get_variable('bs', [1024], initializer=b_init)
        hs = tf.matmul(inputs, ws) + bs
        hs = tf.nn.swish(hs)
        hs = tf.nn.dropout(hs, keep_prob)

        # 2nd hidden layer
        w2 = tf.get_variable('w2', [hs.get_shape()[1], 512], initializer=w_init)
        b2 = tf.get_variable('b2', [512], initializer=b_init)
        h2 = tf.matmul(hs, w2) + b2
        h2 = tf.nn.swish(h2)
        h2 = tf.nn.dropout(h2, keep_prob)

        # 3rd hidden layer
        wt = tf.get_variable('wt', [h2.get_shape()[1], 256], initializer=w_init)
        bt = tf.get_variable('bt', [256], initializer=b_init)
        ht = tf.matmul(h2, wt) + bt
        ht = tf.nn.tanh(ht)
        ht = tf.nn.dropout(ht, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [ht.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        output = tf.matmul(ht, wo) + bo

    return output

# Bernoulli MLP as decoder
def MLP_decoder(z, n_output, keep_prob, reuse=False):
    with tf.variable_scope("MLP_decoder", reuse=reuse):

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], 256], initializer=w_init)
        b0 = tf.get_variable('b0', [256], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1],512], initializer=w_init)
        b1 = tf.get_variable('b1', [512], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.swish(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

         # 2nd hidden layer
        ws = tf.get_variable('ws', [h1.get_shape()[1],1024], initializer=w_init)
        bs = tf.get_variable('bs', [1024], initializer=b_init)
        hs = tf.matmul(h1, ws) + bs
        hs = tf.nn.swish(hs)
        hs = tf.nn.dropout(hs, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [hs.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.nn.sigmoid(tf.matmul(hs, wo) + bo)
    return y

# Discriminator
def discriminator(z, n_output, keep_prob, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], 1024], initializer=w_init)
        b0 = tf.get_variable('b0', [1024], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.swish(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], 1024], initializer=w_init)
        b1 = tf.get_variable('b1', [1024], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.swish(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.matmul(h1, wo) + bo

    return tf.sigmoid(y), y
    
# Gateway
def adversarial_autoencoder(input_data, target_data, x_id, z_sample, z_id, dim_img, dim_z, keep_prob):
    ## Reconstruction Loss
    # encoding
    z = MLP_encoder(input_data, dim_z, keep_prob)

    # decoding
    y = MLP_decoder(z, dim_img, keep_prob)

    # loss
    marginal_likelihood = -tf.reduce_mean(tf.reduce_mean(tf.squared_difference(target_data,y)))

    ## GAN Loss
    z_real = tf.concat([z_sample, z_id],1)
    z_fake = tf.concat([z, x_id],1)
    D_real, D_real_logits = discriminator(z_real, 1024, keep_prob)
    D_fake, D_fake_logits = discriminator(z_fake, 1024,  keep_prob, reuse= True)

    # discriminator loss
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
    D_loss = D_loss_real+D_loss_fake

    # generator loss
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    D_loss = tf.reduce_mean(D_loss)
    G_loss = tf.reduce_mean(G_loss)

    return y, z, -marginal_likelihood, D_loss, G_loss

def decoder(z,dim_img):

    y = MLP_decoder(z, dim_img, 1.0, reuse=True)

    return y


if __name__ == "__main__":
    tf.keras.backend.clear_session()     
    print("loading images... ", end='')
    image_list = glob.glob('images/*.png')

    labels=[]
    images=[]
    split_rate = 0.2

    for i in image_list:
        label = int(i[i.index('_')+1:i.index('.')])-1
        """
        if label >= 3 and label <= 10 :
            continue 
        if label == 11:
            label =3
        if label == 12:
            label =4
        """ # for using only basis activities
        labels.append(label)
        im = Image.open(i)
        images.append(np.array(im))

    train_img = np.array(images).reshape((-1,128*3))
    train_label = np.array(labels).reshape((-1,1))

    print("done")

    xt = train_img
    yt = train_label

    enc = OneHotEncoder()
    enc.fit(yt)
    yt_onehot = enc.transform(yt).toarray()

    tmp_xt = xt

    # params
    epoch = 10000
    learning_rate = 1.46e-3
    dim_z = 3
    feature_num =len(xt[0])
    label_num = 13
    tmp_z = None

    with tf.device('GPU:0'):
        with tf.device('CPU:0'):
            global_step = tf.Variable(0, name='global_step', trainable=False)

        # placeholders
        input_data = tf.placeholder(tf.float32, shape=[None, feature_num], name='input_data')
        target_data = tf.placeholder(tf.float32, shape=[None, feature_num], name='target_data')
        x_id = tf.placeholder(tf.float32, shape=[None, label_num], name='label')

        # dropout
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

        # samples drawn from prior distribution
        z_sample = tf.placeholder(tf.float32, shape=[None, dim_z], name='prior_sample')
        z_id = tf.placeholder(tf.float32, shape=[None, label_num], name='prior_sample_label')

        y, z, neg_marginal_likelihood, D_loss, G_loss = adversarial_autoencoder(input_data, target_data, x_id, z_sample, z_id, feature_num, dim_z, keep_prob)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if "discriminator" in var.name]
        g_vars = [var for var in t_vars if "MLP_encoder" in var.name]
        ae_vars = [var for var in t_vars if "MLP_encoder" or "MLP_decoder" in var.name]

        train_op_ae = tf.train.AdamOptimizer(learning_rate).minimize(neg_marginal_likelihood, var_list=ae_vars)
        train_op_d = tf.train.AdamOptimizer(learning_rate/20).minimize(D_loss, var_list=d_vars)
        train_op_g = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        #sess = tf.Session()

        with tf.device('CPU:0'):
            SAVER_DIR = "AAE_model"
            saver = tf.train.Saver()
            checkpoint_path = os.path.join(SAVER_DIR, "6axis_img")
            ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

        with sess:
            sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

            if ckpt and ckpt.model_checkpoint_path:
                with tf.device('CPU:0'):
                    saver.restore(sess, ckpt.model_checkpoint_path)
            
            for i in range(epoch):
                shuffle = [[x, y] for x, y in zip(xt, yt_onehot)]
                random.shuffle(shuffle)
                xt = [n[0] for n in shuffle]
                yt_onehot = [n[1] for n in shuffle]

                # draw samples from prior distribution 
                '''
                z_id_ = np.random.randint(0, label_num, size=[len(xt)])
                samples = prior.gaussian_mixture(len(xt), dim_z, label_indices=z_id_)
                '''
                
                z_id_ = np.random.randint(0, label_num, size=[len(xt)])  # 'len(xt)' replace 'batch size'
                samples = prior.swiss_roll_3d(len(xt), dim_z, label_indices=z_id_)
                

                '''
                samples, z_id_ = prior.gaussian(len(xt), dim_z, use_label_info=True)
                '''

                z_id_one_hot_vector = np.zeros((len(xt), label_num))
                z_id_one_hot_vector[np.arange(len(xt)), z_id_] = 1

                # reconstruction loss
                _, loss_likelihood = sess.run(
                    (train_op_ae, neg_marginal_likelihood),
                    feed_dict={input_data: xt, target_data: xt, x_id: yt_onehot, z_sample: samples, z_id: z_id_one_hot_vector, keep_prob: 0.9})

                # discriminator loss
                _, d_loss = sess.run(
                    (train_op_d, D_loss),
                    feed_dict={input_data: xt, target_data: xt, x_id: yt_onehot, z_sample: samples, z_id: z_id_one_hot_vector, keep_prob: 0.9})

                # generator loss
                for _ in range(3):
                    _, g_loss = sess.run(
                        (train_op_g, G_loss),
                        feed_dict={input_data: xt, target_data: xt, x_id: yt_onehot, z_sample: samples,
                                   z_id: z_id_one_hot_vector, keep_prob: 0.9})
                tot_loss = loss_likelihood + d_loss + g_loss

                # print cost every 10 epoch
                if i % 500 == 0 :
                    print("epoch %d: L_tot %03.2f L_likelihood %03.2f d_loss %03.2f g_loss %03.2f" % (i, tot_loss, loss_likelihood, d_loss, g_loss))
                    with tf.device('CPU:0'):
                        saver.save(sess, checkpoint_path, global_step=i)
                #print("epoch %d: L_tot %03.2f " % (i, tot_loss))
  
            tmp_z = sess.run(z, feed_dict={input_data: tmp_xt, keep_prob : 1})
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    X = [d[0] for d in tmp_z]
    Y = [d[1] for d in tmp_z]
    Z = [d[2] for d in tmp_z]

    ax.scatter(X,Y,Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    file = open('6_img_aae.csv','a')  

    for i in range(len(yt)+150) :
        rtn = "trash"
        if i < len(yt):
            tmp_list = []
            tmp_list.append(yt[i][0])
            for j in range(dim_z):
                tmp_list.append(tmp_z[i][j])
            rtn = str(tmp_list)[1:-1]
        file.write(str.format(rtn) + "\n")
    
    print("done")