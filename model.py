import tensorflow as tf
from layers import *

# all dimentions are mentioned in terms of an image of dimentions 256 x 256 x 3

# generator with 9 resnet blocks
def create_generator_resnet_9_blocks(input_image_generator, num_filters_generator=64, name="generator"):

    with tf.variable_scope(name):
        # input_image_generator.shape = 256 x 256 x 3
        def create_resnet_block(input_resnet, num_filters, filter_size=3, stride_size=1, name='resnet'):
            padding_size = int((filter_size - 1) / 2)
            output_resnet = tf.pad(input_resnet, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")

            output_resnet = create_convolution_layer(output_resnet, num_filters, filter_size, stride_size, padding='VALID', name=name + '_convolution_1')
            output_resnet = create_instance_normalization_layer(output_resnet, name + '_instance_normalization_1')
            output_resnet = tf.nn.relu(output_resnet)

            output_resnet = tf.pad(output_resnet, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")
            output_resnet = create_convolution_layer(output_resnet, num_filters, filter_size, stride_size, padding='VALID', name=name + '_convolution_2')
            output_resnet = create_instance_normalization_layer(output_resnet, name + '_instance_normalization_2')

            return output_resnet + input_resnet

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        padding_size = 3
        input_image_generator = tf.pad(input_image_generator, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")
        # input_image_generator.shape = 262 x 262 x 3

        convolution_resnet_1 = create_convolution_layer(input_image_generator, num_filters_generator, 7, 1, padding='VALID', name='convolution_resnet_1')
        convolution_resnet_1 = create_instance_normalization_layer(convolution_resnet_1, 'convolution_instance_normalization_1')
        convolution_resnet_1 = tf.nn.relu(convolution_resnet_1)
        # convolution_resnet_1.shape = 256 x 256 x num_filters_generator

        convolution_resnet_2 = create_convolution_layer(convolution_resnet_1, num_filters_generator * 2, 3, 2, name='convolution_resnet_2')
        convolution_resnet_2 = create_instance_normalization_layer(convolution_resnet_2, 'convolution_instance_normalization_2')
        convolution_resnet_2 = tf.nn.relu(convolution_resnet_2)
        # convolution_resnet_2.shape = 128 x 128 x num_filters_generator * 2

        convolution_resnet_3 = create_convolution_layer(convolution_resnet_2, num_filters_generator * 4, 3, 2, name='convolution_resnet_3')
        convolution_resnet_3 = create_instance_normalization_layer(convolution_resnet_3, 'convolution_instance_normalization_3')
        convolution_resnet_3 = tf.nn.relu(convolution_resnet_3)
        # convolution_resnet_2.shape = 64 x 64 x num_filters_generator * 4

        # resnet with 9 blocks
        resnet_block_1 = create_resnet_block(convolution_resnet_3, num_filters_generator * 4, name='resnet_block_1')
        resnet_block_2 = create_resnet_block(resnet_block_1, num_filters_generator * 4, name='resnet_block_2')
        resnet_block_3 = create_resnet_block(resnet_block_2, num_filters_generator * 4, name='resnet_block_3')
        resnet_block_4 = create_resnet_block(resnet_block_3, num_filters_generator * 4, name='resnet_block_4')
        resnet_block_5 = create_resnet_block(resnet_block_4, num_filters_generator * 4, name='resnet_block_5')
        resnet_block_6 = create_resnet_block(resnet_block_5, num_filters_generator * 4, name='resnet_block_6')
        resnet_block_7 = create_resnet_block(resnet_block_6, num_filters_generator * 4, name='resnet_block_7')
        resnet_block_8 = create_resnet_block(resnet_block_7, num_filters_generator * 4, name='resnet_block_8')
        resnet_block_9 = create_resnet_block(resnet_block_8, num_filters_generator * 4, name='resnet_block_9')

        transpose_convolution_resnet_1 = create_transpose_convolution_layer(resnet_block_9, num_filters_generator * 2, 3, 2, name='transpose_convolution_1')
        transpose_convolution_resnet_1 = create_instance_normalization_layer(transpose_convolution_resnet_1, 'transpose_convolution_instance_normalization_1')
        transpose_convolution_resnet_1 = tf.nn.relu(transpose_convolution_resnet_1)
        # transpose_convolution_resnet_1.shape = 128 x 128 x num_filters_generator * 2

        transpose_convolution_resnet_2 = create_transpose_convolution_layer(transpose_convolution_resnet_1, num_filters_generator, 3, 2, name='transpose_convolution_resnet_2')
        transpose_convolution_resnet_2 = create_instance_normalization_layer(transpose_convolution_resnet_2, 'transpose_convolution_instance_normalization_2')
        transpose_convolution_resnet_2 = tf.nn.relu(transpose_convolution_resnet_2)
        # transpose_convolution_resnet_2.shape = 256 x 256 x num_filters_generator

        padding_size = 3
        transpose_convolution_resnet_2 = tf.pad(transpose_convolution_resnet_2, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")
        # transpose_convolution_resnet_2.shape = 262 x 262 x num_filters_generator

        image_channels = 3

        generated_image = create_convolution_layer(transpose_convolution_resnet_2, image_channels, 7, 1, padding='VALID', name='generated_image')
        generated_image = tf.nn.tanh(generated_image)
        # generated_image.shape = 256 x 256 x 3

        return generated_image


# generator with 12 resnet blocks
def create_generator_resnet_12_blocks(input_image_generator, num_filters_generator=64, name="generator"):

    with tf.variable_scope(name):
        # input_image_generator.shape = 256 x 256 x 3
        def create_resnet_block(input_resnet, num_filters, filter_size=3, stride_size=1, name='resnet'):
            padding_size = int((filter_size - 1) / 2)
            output_resnet = tf.pad(input_resnet, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")

            output_resnet = create_convolution_layer(output_resnet, num_filters, filter_size, stride_size, padding='VALID', name=name + '_convolution_1')
            output_resnet = create_instance_normalization_layer(output_resnet, name + '_instance_normalization_1')
            output_resnet = tf.nn.relu(output_resnet)

            output_resnet = tf.pad(output_resnet, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")
            output_resnet = create_convolution_layer(output_resnet, num_filters, filter_size, stride_size, padding='VALID', name=name + '_convolution_2')
            output_resnet = create_instance_normalization_layer(output_resnet, name + '_instance_normalization_2')

            return output_resnet + input_resnet

        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        padding_size = 3
        input_image_generator = tf.pad(input_image_generator, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")
        # input_image_generator.shape = 262 x 262 x 3

        convolution_resnet_1 = create_convolution_layer(input_image_generator, num_filters_generator, 7, 1, padding='VALID', name='convolution_resnet_1')
        convolution_resnet_1 = create_instance_normalization_layer(convolution_resnet_1, 'convolution_instance_normalization_1')
        convolution_resnet_1 = tf.nn.relu(convolution_resnet_1)
        # convolution_resnet_1.shape = 256 x 256 x num_filters_generator

        convolution_resnet_2 = create_convolution_layer(convolution_resnet_1, num_filters_generator * 2, 3, 2, name='convolution_resnet_2')
        convolution_resnet_2 = create_instance_normalization_layer(convolution_resnet_2, 'convolution_instance_normalization_2')
        convolution_resnet_2 = tf.nn.relu(convolution_resnet_2)
        # convolution_resnet_2.shape = 128 x 128 x num_filters_generator * 2

        convolution_resnet_3 = create_convolution_layer(convolution_resnet_2, num_filters_generator * 4, 3, 2, name='convolution_resnet_3')
        convolution_resnet_3 = create_instance_normalization_layer(convolution_resnet_3, 'convolution_instance_normalization_3')
        convolution_resnet_3 = tf.nn.relu(convolution_resnet_3)
        # convolution_resnet_2.shape = 64 x 64 x num_filters_generator * 4

        # resnet with 12 blocks
        resnet_block_1 = create_resnet_block(convolution_resnet_3, num_filters_generator * 4, name='resnet_block_1')
        resnet_block_2 = create_resnet_block(resnet_block_1, num_filters_generator * 4, name='resnet_block_2')
        resnet_block_3 = create_resnet_block(resnet_block_2, num_filters_generator * 4, name='resnet_block_3')
        resnet_block_4 = create_resnet_block(resnet_block_3, num_filters_generator * 4, name='resnet_block_4')
        resnet_block_5 = create_resnet_block(resnet_block_4, num_filters_generator * 4, name='resnet_block_5')
        resnet_block_6 = create_resnet_block(resnet_block_5, num_filters_generator * 4, name='resnet_block_6')
        resnet_block_7 = create_resnet_block(resnet_block_6, num_filters_generator * 4, name='resnet_block_7')
        resnet_block_8 = create_resnet_block(resnet_block_7, num_filters_generator * 4, name='resnet_block_8')
        resnet_block_9 = create_resnet_block(resnet_block_8, num_filters_generator * 4, name='resnet_block_9')
        resnet_block_10 = create_resnet_block(resnet_block_9, num_filters_generator * 4, name='resnet_block_10')
        resnet_block_11 = create_resnet_block(resnet_block_10, num_filters_generator * 4, name='resnet_block_11')
        resnet_block_12 = create_resnet_block(resnet_block_11, num_filters_generator * 4, name='resnet_block_12')

        transpose_convolution_resnet_1 = create_transpose_convolution_layer(resnet_block_12, num_filters_generator * 2, 3, 2, name='transpose_convolution_1')
        transpose_convolution_resnet_1 = create_instance_normalization_layer(transpose_convolution_resnet_1, 'transpose_convolution_instance_normalization_1')
        transpose_convolution_resnet_1 = tf.nn.relu(transpose_convolution_resnet_1)
        # transpose_convolution_resnet_1.shape = 128 x 128 x num_filters_generator * 2

        transpose_convolution_resnet_2 = create_transpose_convolution_layer(transpose_convolution_resnet_1, num_filters_generator, 3, 2, name='transpose_convolution_resnet_2')
        transpose_convolution_resnet_2 = create_instance_normalization_layer(transpose_convolution_resnet_2, 'transpose_convolution_instance_normalization_2')
        transpose_convolution_resnet_2 = tf.nn.relu(transpose_convolution_resnet_2)
        # transpose_convolution_resnet_2.shape = 256 x 256 x num_filters_generator

        padding_size = 3
        transpose_convolution_resnet_2 = tf.pad(transpose_convolution_resnet_2, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")
        # transpose_convolution_resnet_2.shape = 262 x 262 x num_filters_generator

        image_channels = 3

        generated_image = create_convolution_layer(transpose_convolution_resnet_2, image_channels, 7, 1, padding='VALID', name='generated_image')
        generated_image = tf.nn.tanh(generated_image)
        # generated_image.shape = 256 x 256 x 3

        return generated_image



# U-net generator
def create_generator_unet(input_image_generator, num_filters_generator=64, dropout_rate=0.5, name="generator"):
    with tf.variable_scope(name):

        # input_image_generator.shape = 256 x 256 x 3

        convolution_unet_1 = create_convolution_layer(input_image_generator, num_filters_generator, name='convolution_unet_1')
        convolution_unet_1 = create_instance_normalization_layer(convolution_unet_1, 'convolution_unet_instance_normalization_1')
        convolution_unet_1 = leaky_relu(convolution_unet_1)
        # convolution_unet_1.shape = 128 x 128 x num_filters_generator)

        convolution_unet_2 = create_convolution_layer(convolution_unet_1, num_filters_generator*2, name='convolution_unet_2')
        convolution_unet_2 = create_instance_normalization_layer(convolution_unet_2, 'convolution_unet_instance_normalization_2')
        convolution_unet_2 = leaky_relu(convolution_unet_2)
        # convolution_unet_2.shape = 64 x 64 x num_filters_generator * 2)

        convolution_unet_3 = create_convolution_layer(convolution_unet_2, num_filters_generator*4, name='convolution_unet_3')
        convolution_unet_3 = create_instance_normalization_layer(convolution_unet_3, 'convolution_unet_instance_normalization_3')
        convolution_unet_3 = leaky_relu(convolution_unet_3)
        # convolution_unet_3.shape = 32 x 32 x num_filters_generator * 4)

        convolution_unet_4 = create_convolution_layer(convolution_unet_3, num_filters_generator*8, name='convolution_unet_4')
        convolution_unet_4 = create_instance_normalization_layer(convolution_unet_4, 'convolution_unet_instance_normalization_4')
        convolution_unet_4 = leaky_relu(convolution_unet_4)
        # convolution_unet_4.shape = 16 x 16 x num_filters_generator * 8)

        convolution_unet_5 = create_convolution_layer(convolution_unet_4, num_filters_generator*8, name='convolution_unet_5')
        convolution_unet_5 = create_instance_normalization_layer(convolution_unet_5, 'convolution_unet_instance_normalization_5')
        convolution_unet_5 = leaky_relu(convolution_unet_5)
        # convolution_unet_5.shape = 8 x 8 x num_filters_generator * 8)

        convolution_unet_6 = create_convolution_layer(convolution_unet_5, num_filters_generator*8, name='convolution_unet_6')
        convolution_unet_6 = create_instance_normalization_layer(convolution_unet_6, 'convolution_unet_instance_normalization_6')
        convolution_unet_6 = leaky_relu(convolution_unet_6)
        # convolution_unet_6.shape = 4 x 4 x num_filters_generator * 8)

        convolution_unet_7 = create_convolution_layer(convolution_unet_6, num_filters_generator*8, name='convolution_unet_7')
        convolution_unet_7 = create_instance_normalization_layer(convolution_unet_7, 'convolution_unet_instance_normalization_7')
        convolution_unet_7 = leaky_relu(convolution_unet_7)
        # convolution_unet_7.shape = 2 x 2 x num_filters_generator * 8)

        convolution_unet_8 = create_convolution_layer(convolution_unet_7, num_filters_generator*8, name='convolution_unet_8')
        convolution_unet_8 = create_instance_normalization_layer(convolution_unet_8, 'convolution_unet_instance_normalization_8')
        convolution_unet_8 = leaky_relu(convolution_unet_8)
        # convolution_unet_8.shape = 1 x 1 x num_filters_generator * 8)

        transpose_convolution_unet_1 = create_transpose_convolution_layer(convolution_unet_8, num_filters_generator*8, name='transpose_convolution_unet_1')
        transpose_convolution_unet_1 = tf.nn.dropout(transpose_convolution_unet_1, dropout_rate)
        transpose_convolution_unet_1 = create_instance_normalization_layer(transpose_convolution_unet_1, 'transpose_convolution_unet_instance_normalization_1')
        transpose_convolution_unet_1 = tf.concat([transpose_convolution_unet_1, convolution_unet_7], 3)
        transpose_convolution_unet_1 = tf.nn.relu(transpose_convolution_unet_1)
        # transpose_convolution_unet_1.shape = 2 x 2 x num_filters_generator * 8 * 2

        transpose_convolution_unet_2 = create_transpose_convolution_layer(transpose_convolution_unet_1, num_filters_generator*8, name='transpose_convolution_unet_2')
        transpose_convolution_unet_2 = tf.nn.dropout(transpose_convolution_unet_2, dropout_rate)
        transpose_convolution_unet_2 = create_instance_normalization_layer(transpose_convolution_unet_2, 'transpose_convolution_unet_instance_normalization_2')
        transpose_convolution_unet_2 = tf.concat([transpose_convolution_unet_2, convolution_unet_6], 3)
        transpose_convolution_unet_2 = tf.nn.relu(transpose_convolution_unet_2)
        # transpose_convolution_unet_2.shape = 4 x 4 x num_filters_generator * 8 * 2

        transpose_convolution_unet_3 = create_transpose_convolution_layer(transpose_convolution_unet_2, num_filters_generator*8, name='transpose_convolution_unet_3')
        transpose_convolution_unet_3 = tf.nn.dropout(transpose_convolution_unet_3, dropout_rate)
        transpose_convolution_unet_3 = create_instance_normalization_layer(transpose_convolution_unet_3, 'transpose_convolution_unet_instance_normalization_3')
        transpose_convolution_unet_3 = tf.concat([transpose_convolution_unet_3, convolution_unet_5], 3)
        transpose_convolution_unet_3 = tf.nn.relu(transpose_convolution_unet_3)
        # transpose_convolution_unet_3.shape = 8 x 8 x num_filters_generator * 8 * 2

        transpose_convolution_unet_4 = create_transpose_convolution_layer(transpose_convolution_unet_3, num_filters_generator*8, name='transpose_convolution_unet_4')
        transpose_convolution_unet_4 = tf.nn.dropout(transpose_convolution_unet_4, dropout_rate)
        transpose_convolution_unet_4 = create_instance_normalization_layer(transpose_convolution_unet_4, 'transpose_convolution_unet_instance_normalization_4')
        transpose_convolution_unet_4 = tf.concat([transpose_convolution_unet_4, convolution_unet_4], 3)
        transpose_convolution_unet_4 = tf.nn.relu(transpose_convolution_unet_4)
        # transpose_convolution_unet_4.shape = 16 x 16 x num_filters_generator * 8 * 2

        transpose_convolution_unet_5 = create_transpose_convolution_layer(transpose_convolution_unet_4, num_filters_generator*8, name='transpose_convolution_unet_5')
        transpose_convolution_unet_5 = create_instance_normalization_layer(transpose_convolution_unet_5, 'transpose_convolution_unet_instance_normalization_5')
        transpose_convolution_unet_5 = tf.concat([transpose_convolution_unet_5, convolution_unet_3], 3)
        transpose_convolution_unet_5 = tf.nn.relu(transpose_convolution_unet_5)
        # transpose_convolution_unet_5.shape = 32 x 32 x num_filters_generator * 4 * 2

        transpose_convolution_unet_6 = create_transpose_convolution_layer(transpose_convolution_unet_5, num_filters_generator*4, name='transpose_convolution_unet_6')
        transpose_convolution_unet_6 = create_instance_normalization_layer(transpose_convolution_unet_6, 'transpose_convolution_unet_instance_normalization_6')
        transpose_convolution_unet_6 = tf.concat([transpose_convolution_unet_6, convolution_unet_2], 3)
        transpose_convolution_unet_6 = tf.nn.relu(transpose_convolution_unet_6)
        # transpose_convolution_unet_6.shape = 64 x 64 x num_filters_generator * 2 * 2

        transpose_convolution_unet_7 = create_transpose_convolution_layer(transpose_convolution_unet_6, num_filters_generator*2, name='transpose_convolution_unet_7')
        transpose_convolution_unet_7 = create_instance_normalization_layer(transpose_convolution_unet_7, 'transpose_convolution_unet_instance_normalization_7')
        transpose_convolution_unet_7 = tf.concat([transpose_convolution_unet_7, convolution_unet_1], 3)
        transpose_convolution_unet_7 = tf.nn.relu(transpose_convolution_unet_7)
        # transpose_convolution_unet_7.shape = 128 x 128 x num_filters_generator * 1 * 2


        image_channels = 3
        generated_image = create_transpose_convolution_layer(transpose_convolution_unet_7, image_channels, name='transpose_convolution_unet_8')
        # generated_image.shape = 256 x 256 x image_channels

        return tf.nn.tanh(generated_image)



# pixel level discriminator
def create_discriminator_pixel_level(input_image_discriminator, num_filters_discriminator=64, name="discriminator"):

    with tf.variable_scope(name):
        # input_image_discriminator.shape = 256 x 256 x 3
        
        discriminator_convolution_1 = create_convolution_layer(input_image_discriminator, num_filters_discriminator, name='discriminator_convolution_1')
        discriminator_convolution_1 = leaky_relu(discriminator_convolution_1)
        # discriminator_convolution_1.shape = 128 x 128 x num_filters_discriminator

        discriminator_convolution_2 = create_convolution_layer(discriminator_convolution_1, num_filters_discriminator * 2, name='discriminator_convolution_2')
        discriminator_convolution_2 = create_instance_normalization_layer(discriminator_convolution_2, 'discriminator_instance_normalization_1')
        discriminator_convolution_2 = leaky_relu(discriminator_convolution_2)
        # discriminator_convolution_2.shape 64 x 64 x num_filters_discriminator * 2

        discriminator_convolution_3 = create_convolution_layer(discriminator_convolution_2, num_filters_discriminator * 4, name='discriminator_convolution_3')
        discriminator_convolution_3 = create_instance_normalization_layer(discriminator_convolution_3, 'discriminator_instance_normalization_2')
        discriminator_convolution_3 = leaky_relu(discriminator_convolution_3)
        # discriminator_convolution_3.shape 32 x 32 x num_filters_discriminator * 4

        discriminator_convolution_4 = create_convolution_layer(discriminator_convolution_3, num_filters_discriminator * 8, stride_size=1, name='discriminator_convolution_4')
        discriminator_convolution_4 = create_instance_normalization_layer(discriminator_convolution_4, 'discriminator_instance_normalization_3')
        discriminator_convolution_4 = leaky_relu(discriminator_convolution_4)
        # discriminator_convolution_3.shape 32 x 32 x num_filters_discriminator * 8

        discriminator_classification = create_convolution_layer(discriminator_convolution_4, 1, stride_size=1, name='discriminator_classification')
        # discriminator_classification.shape = 32 x 32 x 1
        return discriminator_classification



# patch discriminator
def create_discriminator_patch(input_image_discriminator, num_filters_discriminator=64, patch_size=128, name="discriminator"):

	with tf.variable_scope(name):
	# input_image_discriminator.shape = 256 x 256 x 3
		color_channels = 3
		patch_input = tf.random_crop(input_image_discriminator, [1, patch_size, patch_size, color_channels])
        # patch_input.shape = 128 x 128 x 3

		discriminator_convolution_1 = create_convolution_layer(patch_input, num_filters_discriminator, name='discriminator_convolution_1')
		discriminator_convolution_1 = leaky_relu(discriminator_convolution_1)
		# discriminator_convolution_1.shape = 64 x 64 x num_filters_discriminator

		discriminator_convolution_2 = create_convolution_layer(discriminator_convolution_1, num_filters_discriminator * 2, name='discriminator_convolution_2')
		discriminator_convolution_2 = create_instance_normalization_layer(discriminator_convolution_2, 'discriminator_instance_normalization_1')
		discriminator_convolution_2 = leaky_relu(discriminator_convolution_2)
		# discriminator_convolution_2.shape = 32 x 32 x num_filters_discriminator * 2

		discriminator_convolution_3 = create_convolution_layer(discriminator_convolution_2, num_filters_discriminator * 4, name='discriminator_convolution_3')
		discriminator_convolution_3 = create_instance_normalization_layer(discriminator_convolution_3, 'discriminator_instance_normalization_2')
		discriminator_convolution_3 = leaky_relu(discriminator_convolution_3)
		# discriminator_convolution_3.shape = 16 x 16 x num_filters_discriminator * 4

		discriminator_convolution_4 = create_convolution_layer(discriminator_convolution_3, num_filters_discriminator * 8, stride_size=1, name='discriminator_convolution_4')
		discriminator_convolution_4 = create_instance_normalization_layer(discriminator_convolution_4, 'discriminator_instance_normalization_3')
		discriminator_convolution_4 = leaky_relu(discriminator_convolution_4)
		# discriminator_convolution_3.shape = 16 x 16 x num_filters_discriminator * 8

		discriminator_classification = create_convolution_layer(discriminator_convolution_4, 1, stride_size=1, name='discriminator_classification')
		# discriminator_classification.shape = 16 x 16 x 1
		return discriminator_classification