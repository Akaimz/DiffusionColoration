#%%
import tensorflow as tf


# Links: 
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
# https://github.com/emilwallner/Coloring-greyscale-images/blob/master/Alpha-version/alpha_version_notebook.ipynb
# https://towardsdatascience.com/u-nets-with-resnet-encoders-and-cross-connections-d8ba94125a2c
# https://github.com/richzhang/colorization/tree/master
# https://keras.io/examples/generative/ddim/

# https://www.kaggle.com/code/basu369victor/image-colorization-basic-implementation-with-cnn

# Diffusion Model
# https://medium.com/@erwannmillon/color-diffusion-colorizing-black-and-white-images-with-diffusion-models-269828f71c81
# https://github.com/ErwannMillon/Color-diffusion/blob/main/dataset.py
# https://dl.acm.org/doi/fullHtml/10.1145/3528233.3530757

#simple diffusion model:
#https://tree.rocks/make-diffusion-model-from-scratch-easy-way-to-implement-quick-diffusion-model-e60d18fd0f2e

#%%

# Unet 


def downsampling_block(x_input, units):

    #3x3 maxpool with ReLU activation 2 times
    #1
    h = tf.keras.layers.Conv2D(units, kernel_size=3, padding='SAME', use_bias=False, activation = "relu")(x_input)
    h = tf.keras.layers.BatchNormalization()(h)
    
    #2
    h = tf.keras.layers.Conv2D(units, kernel_size=3, padding='SAME', use_bias=False,  activation = "relu")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
    #3
    
    # max pooling
    h = tf.keras.layers.MaxPooling2D(pool_size=2, strides=4)(h)

    
    return h



def upsample_block(x, conv_features, units):
    # upsample
    h = tf.keras.layers.Conv2DTranspose(units, 3, 2, padding="same")(x)
    # concatenate
    h = tf.keras.layers.concatenate([x, conv_features])
    # dropout
    h = tf.keras.layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    h = tf.keras.layers.Conv2D(units, kernel_size=3, padding='SAME', use_bias=False, activation = "relu")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Conv2D(units, kernel_size=3, padding='SAME', use_bias=False, activation = "relu")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    return x



def basic_residual_block(x_input, in_units, units, stride):
    h = tf.keras.layers.Conv2D(units, kernel_size=3, strides=stride, padding='SAME', use_bias=False)(x_input)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
    h = tf.keras.layers.Conv2D(units, kernel_size=3, strides=1, padding='SAME', use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    if stride != 1 or in_units != units:
        x_input = tf.keras.layers.Conv2D(units, kernel_size=1, strides=stride, padding='VALID', use_bias=False)(x_input)
        x_input = tf.keras.layers.BatchNormalization()(x_input)
    h = tf.keras.layers.Add()([h, x_input])
    h = tf.keras.layers.Activation(tf.keras.activations.relu)(h)
    return h