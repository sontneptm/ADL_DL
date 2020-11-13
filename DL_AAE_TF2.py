import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import ADL_PF as prior
from ADL_util import gradient_penalty
from sklearn.preprocessing import OneHotEncoder
from functools import partial
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import models

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# load data
whole_data = np.loadtxt("6axis_raw.csv",delimiter=',', dtype=np.float32)
y_data = whole_data[:,:1]
x_data = whole_data[:,1:]
x_data = (x_data-x_data.min())/(x_data.max()-x_data.min())

#enc = OneHotEncoder()
#enc.fit(y_data)
#yt_onehot = enc.transform(y_data).toarray()
y_data_back = y_data
#y_data = yt_onehot
y_data = y_data.reshape(-1,1)

# hyper params
SPLIT_RATE = 0.2
EPOCHS = 1000
DATA_LENGTH = len(x_data[0])
AE_LR = 1.46e-3
GEN_LR = 1.46e-3
DSC_LR = GEN_LR/20.0
VALID_STEP = 20
CLASS_NUM = 13
LATENT_DIM = 3
SAVE_FREQ = 100

class Encoder(object):
    def __init__(self):
        self.model()

    def model(self):
        model = tf.keras.Sequential(name='Encoder')

        # Layer 1
        model.add(layers.Dense(1024, activation=tf.nn.swish, input_shape=[DATA_LENGTH]))
        model.add(layers.Dense(512, activation=tf.nn.swish))
        model.add(layers.Dense(256, activation=tf.nn.swish))
        model.add(layers.Dense(LATENT_DIM))

        return model
# end of Encoder class

class Decoder(object):
    def __init__(self):
        self.model()

    def model(self):
        model = tf.keras.Sequential(name='Decoder')

        # Layer 1
        model.add(layers.Dense(256, activation=tf.nn.swish, input_shape=[LATENT_DIM]))
        model.add(layers.Dense(512, activation=tf.nn.swish))
        model.add(layers.Dense(1024, activation=tf.nn.swish))
        model.add(layers.Dense(DATA_LENGTH, activation='sigmoid'))

        return model
# end of Decoder class

class Discriminator(object):
    def __init__(self):
        pass

    def model(self):
        model = tf.keras.Sequential(name='Discriminator')

        # Layer 1
        model.add(layers.Dense(1024, activation=tf.nn.swish, input_shape=[LATENT_DIM+1]))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(512, activation=tf.nn.swish))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1))
    
        return model
# end of Discriminator class

def train():
    encoded_data=None
    decoded_data=None

    # build models
    enc = Encoder().model()
    dec = Decoder().model()
    dsc = Discriminator().model()

    # set optimizers
    opt_ae = tf.keras.optimizers.Adam(learning_rate=AE_LR, beta_1=0.5, beta_2=0.999, epsilon=0.01)
    opt_gen = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.5, beta_2=0.999, epsilon=0.01)
    opt_dsc = tf.keras.optimizers.Adam(learning_rate=DSC_LR, beta_1=0.5, beta_2=0.999, epsilon=0.01)

    # Set trainable variables
    var_ae = enc.trainable_variables + dec.trainable_variables
    var_gen = enc.trainable_variables
    var_dsc = dsc.trainable_variables

    # Check point
    check_point_dir = os.path.join(os.getcwd(), 'training_checkpoints')

    graph_path = os.path.join(os.getcwd(), 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(os.getcwd(), 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    check_point_prefix = os.path.join(check_point_dir, 'aae')

    enc_name = 'aae_enc'
    dec_name = 'aae_dec'
    dsc_name = 'aae_dsc'

    graph = 'aae'

    check_point = tf.train.Checkpoint(opt_gen=opt_gen, opt_dcs=opt_dsc, opt_ae=opt_ae, encoder=enc, decoder=dec, discriminator=dsc)
    ckpt_manager = tf.train.CheckpointManager(check_point, check_point_dir, max_to_keep=5,checkpoint_name=check_point_prefix)

    # define generator training step
    def gen_training_step(x, y):
        with tf.GradientTape() as gen_tape:
            z_gen = enc(x, training = True)
            z_gen_input = tf.concat([z_gen, y], 1, name='z_gen_input')
            
            dsc_fake = dsc(z_gen_input, training=True)
            loss_gen = -tf.reduce_mean(dsc_fake)
        grad_gen = gen_tape.gradient(loss_gen, var_gen)
        opt_gen.apply_gradients(zip(grad_gen, var_gen))

    #define AE training step
    def training_step(x, y, z, z_id):
        with tf.GradientTape() as ae_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as dsc_tape:
            z_real = z
            z_gen = enc(np.asmatrix(x), training=True)

            z_input = tf.concat([z_real,y], 1, name='z_input')
            z_gen_input = tf.concat([z_gen,z_id], 1, name='z_gen_input')
            z_gen_input_for_dec = tf.concat([z_gen], 1, name='z_gen_input')

            x_bar = dec(z_gen_input_for_dec, training=True)

            dsc_real = dsc(z_input, training=True)
            dsc_fake = dsc(z_gen_input, training=True)

            real_loss = -tf.reduce_mean(dsc_real)
            fake_loss = tf.reduce_mean(dsc_fake)
            gp = gradient_penalty(partial(dsc, training=True), z_input, z_gen_input)

            loss_gen = -tf.reduce_mean(dsc_fake)
            loss_dsc = (real_loss + fake_loss) + gp * 0.1
            loss_ae = tf.reduce_mean(tf.abs(tf.subtract(x, x_bar)))
        
        grad_ae = ae_tape.gradient(loss_ae, var_ae)
        grad_gen = gen_tape.gradient(loss_gen, var_gen)
        grad_dsc = dsc_tape.gradient(loss_dsc, var_dsc)

        opt_ae.apply_gradients(zip(grad_ae, var_ae))
        opt_gen.apply_gradients(zip(grad_gen, var_gen))
        opt_dsc.apply_gradients(zip(grad_dsc, var_dsc))

    #define validation step
    def validation_step(x, y, z, z_id):
        z_real = z
        z_gen = enc(np.asmatrix(x), training=False)

        z_input = tf.concat([z_real,y], 1, name='z_input')
        z_gen_input = tf.concat([z_gen, z_id], 1, name='z_gen_input')
        z_input_for_dec = tf.concat([z_real], 1, name='z_input')
        z_gen_input_for_dec = tf.concat([z_gen], 1, name='z_gen_input')

        x_bar = dec(z_gen_input_for_dec, training=False)
        x_gen = dec(z_input_for_dec, training=False)

        dsc_real = dsc(z_input, training=False)
        dsc_fake = dsc(z_gen_input, training=False)

        real_loss = -tf.reduce_mean(dsc_real)
        fake_loss = tf.reduce_mean(dsc_fake)
        gp = gradient_penalty(partial(dsc, training=False), z_input, z_gen_input)

        loss_gen = -tf.reduce_mean(dsc_fake)
        loss_dsc = (real_loss + fake_loss) + gp * 0.1
        loss_ae = tf.reduce_mean(tf.abs(tf.subtract(x, x_bar)))

        return x_gen, x_bar, loss_dsc.numpy(), loss_gen.numpy(), loss_ae.numpy(), (fake_loss.numpy()-real_loss.numpy())

    # do TRAIN

    start_time = time.time()
    for epoch in range(EPOCHS):
        # train AAE
        z_id = np.random.randint(0, CLASS_NUM, size=[len(x_data)])  # 'len(xt)' replace 'batch size'
        z_id = z_id.reshape(-1,1)
        samples = prior.swiss_roll_3d(len(x_data), LATENT_DIM, label_indices=z_id)

        #z_id_one_hot_vector = np.zeros((len(x_data), CLASS_NUM))
        #z_id_one_hot_vector[np.arange(len(x_data)), z_id] = 1

        num_train = 0
        if num_train % 2 == 0:
            training_step(x_data ,y_data, samples, z_id)
        else :
            gen_training_step(x_data,y_data)
            gen_training_step(x_data,y_data)
        num_train +=1
        
        # validation
        num_valid = 0
        val_loss_dsc, val_loss_gen, val_loss_ae, val_was_x = [],[],[],[]
        
        x_gen, x_bar, loss_dsc, loss_gen, loss_ae, was_x = validation_step(x_data, y_data, samples, z_id)

        val_loss_dsc.append(loss_dsc)
        val_loss_gen.append(loss_gen)
        val_loss_ae.append(loss_ae)
        val_was_x.append(was_x)

        num_valid += 1

        if num_valid > VALID_STEP:
            break
        
        elapsed_time = (time.time() - start_time) /60.
        val_loss_ae = np.mean(np.reshape(val_loss_ae, (-1)))
        val_loss_dsc = np.mean(np.reshape(val_loss_dsc, (-1)))
        val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))
        val_was_x = np.mean(np.reshape(val_was_x, (-1)))

        print("[Epoch: {:05d}] {:.01f}m.\tdsc: {:.6f}\tgen: {:.6f}\tae: {:.6f}\tw_x: {:.6f}".format(epoch, elapsed_time, val_loss_dsc, val_loss_gen, val_loss_ae, val_was_x))
        
        if epoch % SAVE_FREQ == 0 and epoch > 1:
            ckpt_manager.save(checkpoint_number=epoch)

        if epoch == EPOCHS-1 :
            encoded_data = enc(x_data, training=False)
            encoded_input = tf.concat([encoded_data], 1, name='z_gen_input')
            decoded_data = dec(encoded_input)

    save_message = "\tSave model: End of training"

    enc.save_weights(os.path.join(model_path, enc_name))
    dec.save_weights(os.path.join(model_path, dec_name))
    dsc.save_weights(os.path.join(model_path, dsc_name))

    # 6-3. Report
    print("[Epoch: {:04d}] {:.01f} min.".format(EPOCHS, elapsed_time))
    print(save_message)

    x_axis = range(len(x_data[0]))
    encoded_val_data = encoded_data.numpy()
    decoded_val_data = decoded_data.numpy()

    file = open('6axis_aae_tf2.csv','a')

    for i in range(len(y_data_back)):
        tmp_list = []
        tmp_list.append(y_data_back[i][0])
        for j in range(LATENT_DIM):
            tmp_list.append(encoded_val_data[i][j])
        rtn = str(tmp_list)[1:-1]
        file.write(str.format(rtn) + "\n")

    plt.plot(x_axis, decoded_val_data[5])
    plt.plot(x_axis, x_data[5])
    plt.show()
    
# end of train() method

if __name__ == "__main__":
    train()
    