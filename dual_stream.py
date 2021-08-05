import numpy as np
import cv2 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Add, Maximum, Dense, Softmax
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.utils import plot_model

class Dual_stream():
    def __init__(self, classes=10):
        self.c = classes

    # @tf.function
    def conv(self, filters, kernel, pool, padding='same', s_pool=2, inp_size=(80, 80, 3)):
        # img_inputs = Input(shape=inp_size)
        model = Sequential()
        # otpt = Conv2D(filters=filters, kernel_size=kernel, padding=padding, activation='relu')(img_inputs)
        model.add(Conv2D(filters=filters, kernel_size=kernel, padding=padding, activation='relu'))
        if pool:
            # otpt = MaxPool2D(pool_size=(2,2), strides=s_pool)(otpt)
            model.add(MaxPool2D(pool_size=(2,2), strides=s_pool))
        # return Model(inputs=img_inputs, outputs=otpt)
        return model
    
    # @tf.function
    def construct(self, inp_size=(80,80,3)):
        # Channel 1
        inp1 = Input(shape= inp_size)
        s1_conv1 = self.conv(filters=64,  kernel=(2,2), pool=True,  padding='same', inp_size=inp_size, s_pool=2)(inp1)
        s1_conv2 = self.conv(filters=128, kernel=(2,2), pool=True,  padding='same', inp_size=s1_conv1.shape, s_pool=2)(s1_conv1)
        s1_conv3 = self.conv(filters=256, kernel=(2,2), pool=False, padding='same', inp_size=s1_conv2.shape)(s1_conv2)
        s1_conv4 = self.conv(filters=256, kernel=(2,2), pool=True,  padding='same', inp_size=s1_conv3.shape, s_pool=2)(s1_conv3)
        s1_conv5 = self.conv(filters=512, kernel=(2,2), pool=False, padding='same', inp_size=s1_conv4.shape)(s1_conv4)
        s1_conv6 = self.conv(filters=512, kernel=(2,2), pool=True,  padding='same', inp_size=s1_conv5.shape, s_pool=2)(s1_conv5)
        s1_conv7 = self.conv(filters=512, kernel=(2,2), pool=False, padding='same', inp_size=s1_conv6.shape)(s1_conv6)
        s1_conv8 = self.conv(filters=512, kernel=(2,2), pool=False, padding='same', inp_size=s1_conv7.shape)(s1_conv7)
        s1_flat  = Flatten()(s1_conv8)

        # Channel 2
        inp2 = Input(shape= inp_size)
        s2_conv1 = self.conv(filters=64,  kernel=(2,2), pool=True,  padding='same', inp_size=inp_size, s_pool=2)(inp2)
        s2_conv2 = self.conv(filters=128, kernel=(2,2), pool=True,  padding='same', inp_size=s2_conv1.shape, s_pool=2)(s2_conv1)
        s2_conv3 = self.conv(filters=256, kernel=(2,2), pool=False, padding='same', inp_size=s2_conv2.shape)(s2_conv2)
        s2_conv4 = self.conv(filters=256, kernel=(2,2), pool=True,  padding='same', inp_size=s2_conv3.shape, s_pool=2)(s2_conv3)
        s2_conv5 = self.conv(filters=512, kernel=(2,2), pool=False, padding='same', inp_size=s2_conv4.shape)(s2_conv4)
        s2_conv6 = self.conv(filters=512, kernel=(2,2), pool=True,  padding='same', inp_size=s2_conv5.shape, s_pool=2)(s2_conv5)
        s2_conv7 = self.conv(filters=512, kernel=(2,2), pool=False, padding='same', inp_size=s2_conv6.shape)(s2_conv6)
        s2_conv8 = self.conv(filters=512, kernel=(2,2), pool=False, padding='same', inp_size=s2_conv7.shape)(s2_conv7)
        s2_flat  = Flatten()(s2_conv8)

        # Common
        z_sum = Add()([s1_flat, s2_flat])
        z_max = Maximum()([s1_flat, s2_flat])

        # z max channel
        max_d1 = Dense(units=4096, activation='relu')(z_max)
        max_d2 = Dense(units=4096, activation='relu')(max_d1)
        y_max = Dense(units=self.c, activation='relu')(max_d2)
        # o_max = Softmax()(y_max)

        # z max channel
        sum_d1 = Dense(units=4096, activation='relu')(z_sum)
        sum_d2 = Dense(units=4096, activation='relu')(sum_d1)
        y_sum = Dense(units=self.c, activation='relu')(sum_d2)
        # o_sum = Softmax()(y_sum)

        # Final output 
        return Model(inputs=[inp1, inp2], outputs=[y_max, y_sum])

    # @tf.function
    def full_model(self, input=np.zeros((80,80,3))):

        inp1 = Input(shape=input.shape)
        inp2 = Input(shape=input.shape)
        inp3 = Input(shape=input.shape)
        inp4 = Input(shape=input.shape)

        # Right ocular stream
        r_ocular = self.construct(inp_size=input.shape)
        x_r = r_ocular([inp1, inp2])

        # Left Ocular stream
        l_ocular = self.construct(inp_size=input.shape)
        x_l = l_ocular([inp3, inp4])
        return Model(inputs=[(inp1, inp2), (inp3, inp4)], outputs = [x_r, x_l])

    @tf.function
    def foreward(self, inputs):
        # Right ocular stream
        r_ocular = self.construct(inp_size=inputs[0].shape)
        x_r = r_ocular([inputs[0], inputs[1]])

        # Left Ocular stream
        l_ocular = self.construct(inp_size=inputs[0].shape)
        x_l = l_ocular([inputs[2], inputs[3]])
        return [x_r, x_l]