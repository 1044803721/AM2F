from keras.models import Model
from keras.layers import Dense, Input, Conv1D, concatenate, Dropout, \
    LSTM, Activation, multiply, add, Softmax,Reshape

from models.modules import resnet
import keras.backend as K
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.layers import Lambda
import keras
from keras import layers

import numpy as np
from MCB import MCB


class AM2F:

    def simple_cnn(self, X_input, net_id):
        X = Conv1D(filters=self.cnn_args[0], kernel_size=self.cnn_args[1],
                   kernel_initializer="glorot_uniform", name="simple_conv1_%s_" % net_id)(X_input)
        X = Activation("relu")(X)
        return X

    def __init__(self, window_size, model_args, data_type):
        self.window_size = window_size
        self.lstm_args = model_args["lstm"]
        self.cnn_args = model_args["cnn"]
        self.resnet_args = model_args["resnet"]
        self.attention_args = model_args["attention"]
        self.fc_args = model_args["fc"]
        self.dropout_args = model_args["dropout"]
        self.data_type = data_type

    def build_model(self):
        gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z, pressure = self.input_layer()
        gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn, mag_y_cnn, mag_z_cnn, pressure_cnn = self.residual_layer(
            gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z, pressure)
        all_resnet = self.cnn_layer(gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn,
                                    mag_y_cnn, mag_z_cnn, pressure_cnn)
        lstm = self.lstm_layer(all_resnet)
        lstm = self.attention_layer(lstm)
        output = self.mlp_layer(lstm)
        model = Model(inputs=[
            gyr_x, gyr_y, gyr_z,
            lacc_x, lacc_y, lacc_z,
            mag_x, mag_y, mag_z, pressure
        ],
            outputs=[output])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def mlp_layer(self, lstm):
        fc = Dense(self.fc_args[0], activation='relu', kernel_initializer='truncated_normal')(lstm)
        fc = Dropout(self.dropout_args)(fc)
        fc = Dense(self.fc_args[1], activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(self.dropout_args)(fc)
        fc = Dense(self.fc_args[2], activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(self.dropout_args)(fc)
        fc = Dense(self.fc_args[3], activation='relu', kernel_initializer='truncated_normal')(fc)
        fc = Dropout(self.dropout_args)(fc)
        print(fc, '////')
        output = Dense(self.fc_args[4], activation='softmax', name='output')(fc)
        return output

    def attention_layer(self, lstm):
        dense1 = Dense(self.attention_args[0], activation="softmax")(lstm)
        dense2 = Dense(self.attention_args[1], activation="softmax")(dense1)
        lstm = multiply([lstm, dense2])
        return lstm

    def lstm_layer(self, all_resnet):
        lstm = LSTM(self.lstm_args[0], input_shape=(self.lstm_args[1], self.lstm_args[2]), activation='tanh',
                    dropout=self.dropout_args, recurrent_dropout=self.dropout_args)(all_resnet)
        print(lstm,"@@@")
        return lstm

    def cnn_layer(self, gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn, mag_y_cnn,
                  mag_z_cnn, pressure_cnn):
        get_lambda2 = lambda x: MCB(x[0], x[1])
        get_lambda_layer2 = Lambda(get_lambda2)
        print(get_lambda_layer2([lacc_x_cnn, lacc_y_cnn]))
        print('///')
        a = get_lambda_layer2([lacc_x_cnn, lacc_y_cnn])
        b = get_lambda_layer2([lacc_x_cnn, lacc_z_cnn])
        c = get_lambda_layer2([lacc_y_cnn, lacc_z_cnn])
        d2 = Dense(512, activation='tanh')(a)
        e2 = Dense(512, activation='tanh')(b)
        f2 = Dense(512, activation='tanh')(c)
        g2 = add(([d2, e2, f2]))
        h2 = Softmax(axis=1)(g2)
        h21 = Lambda(lambda x: x[:, :, 0:1])(h2)
        h22 = Lambda(lambda x: x[:, :, 1:2])(h2)
        h23 = Lambda(lambda x: x[:, :, 2:3])(h2)
        a_w = add([h21,h22])
        b_w = add([h21, h23])
        c_w = add([h22, h23])
        a_w = Lambda(lambda x: x / 2)(a_w)
        b_w = Lambda(lambda x: x / 2)(b_w)
        c_w = Lambda(lambda x: x / 2)(c_w)
        x_w = multiply([lacc_x_cnn, a_w])
        y_w = multiply([lacc_y_cnn, b_w ])
        z_w = multiply([lacc_z_cnn, c_w ])
        concat_lacc = add([x_w,y_w,z_w])


        d = get_lambda_layer2([gyr_x_cnn, gyr_y_cnn])
        e = get_lambda_layer2([gyr_x_cnn, gyr_z_cnn])
        f = get_lambda_layer2([gyr_y_cnn, gyr_z_cnn])
        d1 = Dense(128, activation='tanh')(d)
        e1 = Dense(128, activation='tanh')(e)
        f1 = Dense(128, activation='tanh')(f)
        g1 = add(([d1, e1, f1]))
        h1 = Softmax(axis=1)(g1)
        h11 = Lambda(lambda x: x[:, :, 0:1])(h1)
        h12 = Lambda(lambda x: x[:, :, 1:2])(h1)
        h13 = Lambda(lambda x: x[:, :, 2:3])(h1)
        a_w1 = add([h11, h12])
        b_w1 = add([h11, h13])
        c_w1 = add([h12, h13])
        a_w1 = Lambda(lambda x: x / 2)(a_w1)
        b_w1 = Lambda(lambda x: x / 2)(b_w1)
        c_w1 = Lambda(lambda x: x / 2)(c_w1)
        x_w1 = multiply([gyr_x_cnn, a_w1])
        y_w1 = multiply([gyr_y_cnn, b_w1])
        z_w1 = multiply([gyr_z_cnn, c_w1])
        concat_gyr = add([x_w1, y_w1, z_w1])


        g = get_lambda_layer2([mag_x_cnn, mag_y_cnn])
        h = get_lambda_layer2([mag_x_cnn, mag_z_cnn])
        i = get_lambda_layer2([mag_y_cnn, mag_z_cnn])
        d3 = Dense(128, activation='tanh')(g)
        e3 = Dense(128, activation='tanh')(h)
        f3 = Dense(128, activation='tanh')(i)
        g3 = add(([d3, e3, f3]))
        h3 = Softmax(axis=1)(g3)
        h31 = Lambda(lambda x: x[:, :, 0:1])(h3)
        h32 = Lambda(lambda x: x[:, :, 1:2])(h3)
        h33 = Lambda(lambda x: x[:, :, 2:3])(h3)
        a_w3 = add([h31, h32])
        b_w3 = add([h31, h33])
        c_w3 = add([h32, h33])
        a_w3 = Lambda(lambda x: x / 2)(a_w3)
        b_w3 = Lambda(lambda x: x / 2)(b_w3)
        c_w3 = Lambda(lambda x: x / 2)(c_w3)
        x_w3 = multiply([gyr_x_cnn, a_w3])
        y_w3 = multiply([gyr_y_cnn, b_w3])
        z_w3 = multiply([gyr_z_cnn, c_w3])
        concat_mag = add([x_w3, y_w3, z_w3])



        concat_lacc_resnet = self.simple_cnn(concat_lacc, "concat_lacc")
        concat_gyr_resnet = self.simple_cnn(concat_gyr, "concat_gyr")
        print(concat_gyr_resnet)
        concat_mag_resnet = self.simple_cnn(concat_mag, "concat_mag")
        concat_pressure_resnet = self.simple_cnn(pressure_cnn, "concat_pressure")
        all_resnet = concatenate(
            [concat_lacc_resnet, concat_gyr_resnet, concat_mag_resnet, concat_pressure_resnet])
        print(all_resnet)
        return all_resnet

    def residual_layer(self, gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z, pressure):
        lacc_x_cnn = resnet.res_net(lacc_x, "single_lacc_x", self.resnet_args)
        lacc_y_cnn = resnet.res_net(lacc_y, "single_lacc_y", self.resnet_args)
        lacc_z_cnn = resnet.res_net(lacc_z, "single_lacc_z", self.resnet_args)
        gyr_x_cnn = resnet.res_net(gyr_x, "single_gyr_x", self.resnet_args)
        gyr_y_cnn = resnet.res_net(gyr_y, "single_gyr_y", self.resnet_args)
        gyr_z_cnn = resnet.res_net(gyr_z, "single_gyr_z", self.resnet_args)
        mag_x_cnn = resnet.res_net(mag_x, "single_mag_x", self.resnet_args)
        mag_y_cnn = resnet.res_net(mag_y, "single_mag_y", self.resnet_args)
        mag_z_cnn = resnet.res_net(mag_z, "single_mag_z", self.resnet_args)
        pressure_cnn = resnet.res_net(pressure, "single_pressure", self.resnet_args)
        return gyr_x_cnn, gyr_y_cnn, gyr_z_cnn, lacc_x_cnn, lacc_y_cnn, lacc_z_cnn, mag_x_cnn, mag_y_cnn, mag_z_cnn, pressure_cnn

    def input_layer(self):
        lacc_x = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccx_input')
        lacc_y = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccy_input')
        lacc_z = Input(shape=(self.window_size, 1),
                       dtype='float32', name='laccz_input')
        gyr_x = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyrx_input')
        gyr_y = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyry_input')
        gyr_z = Input(shape=(self.window_size, 1),
                      dtype='float32', name='gyrz_input')
        mag_x = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magx_input')
        mag_y = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magy_input')
        mag_z = Input(shape=(self.window_size, 1),
                      dtype='float32', name='magz_input')
        pressure = Input(shape=(self.window_size, 1),
                         dtype='float32', name='pres_input')
        return gyr_x, gyr_y, gyr_z, lacc_x, lacc_y, lacc_z, mag_x, mag_y, mag_z, pressure

