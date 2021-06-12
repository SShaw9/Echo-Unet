import tensorflow as tf
from tensorflow import keras
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Input, UpSampling2D
from keras.models import Model


def build_unet():
    """ Build a Unet based neural architecture that uses pretrained weights from ImageNet """
    inputs = Input((112, 112, 3), name='input_image')

    # Pretrained encoder block
    base_encoder = tf.keras.applications.MobileNetV2(input_tensor=inputs,
                                                     weights='imagenet',
                                                     include_top=False,
                                                     alpha=0.35)
    # base_encoder.summary()
    skip_names = ['input_image',
                  'block_1_expand_relu',   # 56x56
                  'block_3_expand_relu',   # 28x28
                  'block_6_expand_relu']   # 14x14

    encoder_output = base_encoder.get_layer('block_13_expand_relu').output # 7x7

    filters = [64, 128, 256, 512]
    x = encoder_output

    for i in range(1, len(filters)+1, 1):
        x_skip = base_encoder.get_layer(skip_names[-i]).output
        x = UpSampling2D((2,2))(x)
        x = Concatenate()([x, x, x_skip])

        x = Conv2D(filters[-i], (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters[-i], (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(1, (1,1), padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x)

    return model

