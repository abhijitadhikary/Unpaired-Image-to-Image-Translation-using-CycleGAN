import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import copy

# create forward convolution layer
def create_convolution_layer(input_image, num_filters, filter_size=4, stride_size=2, stddev=0.02, padding='SAME', name="convolution_layer"):
    with tf.variable_scope(name):
        return slim.conv2d(input_image, 
                           num_filters, 
                           filter_size, 
                           stride_size, 
                           padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)

# create transpose convolution layer
def create_transpose_convolution_layer(input_image, num_filters, filter_size=4, stride_size=2, stddev=0.02, name="transpose_convolution_layer"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_image, 
                                     num_filters, 
                                     filter_size, 
                                     stride_size, 
                                     padding='SAME', 
                                     activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)

# create instance normalization layer
def create_instance_normalization_layer(input_image, name="instance_normalization_layer"):
    with tf.variable_scope(name):
        depth = input_image.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_image, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_image - mean) * inv
        return scale*normalized + offset

# create leaky ReLU layer
def leaky_relu(input, leak=0.2, name="leaky_relu"):
    with tf.variable_scope(name):
        return tf.maximum(input, leak * input)


# class for creating and storing the image pool (returns one image from each dataset)
class ImagePool(object):
    def __init__(self, max_pool_size=50):
        self.max_pool_size = max_pool_size
        self.num_images_in_pool = 0
        self.images = []

    def __call__(self, image):

        # if pool is empty return the current image
        if self.max_pool_size <= 0:
            return image

        # if image pool is not full yet, add the image to the pool, return the image
        if self.num_images_in_pool < self.max_pool_size:
            self.images.append(image)
            self.num_images_in_pool += 1
            return image

        # if image pool is full, random number is greater than 0.5, return any random image from the image 
        if np.random.rand() > 0.5:
            random_index = int(np.random.rand() * self.max_pool_size)
            temp_image_A = copy.copy(self.images[random_index])[0]
            self.images[random_index][0] = image[0]

            random_index = int(np.random.rand() * self.max_pool_size)
            temp_image_B = copy.copy(self.images[random_index])[1]
            self.images[random_index][1] = image[1]
            return [temp_image_A, temp_image_B]
        # else return the current images
        else:
            return image