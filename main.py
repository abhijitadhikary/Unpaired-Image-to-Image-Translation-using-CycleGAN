import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import time
import random
from layers import *
from model import *
import datetime
# from inception_score import *
# from fid import *

# ignore CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# specify the dataset and running mode
dataset_name = "horse2zebra"
training_condition = True
test_condition = False
restore_condition = False
# run_inception_condition = True
restore_dataset_folder = '26-04-2019__11h-19m'

# testing conditions
total_test_images = 100
test_A_to_B = True
test_B_to_A = False
load_custom_test_dataset = False
custom_test_dataset = 'h2z'



# specify the partial trained folder
if restore_condition == True or test_condition == True:
    timeStamp = restore_dataset_folder
# get the current date and time, use the information to create the training folder
else:
    currentDT = datetime.datetime.now()
    timeStamp = currentDT.strftime("%d-%m-%Y__%Hh-%Mm")

# i/0 parameters
output_path = "./output/{}/{}".format(dataset_name, timeStamp)
checkpoint_path = "./{}/checkpoints/".format(output_path)
images_path = "{}/images".format(output_path)
save_training_images_condition = True
num_train_images_to_save = 10
# path for saving the test images
test_path = '{}/test_images'.format(output_path)


# specify the image dimentions
image_height = 256
image_width = 256
image_channels = 3
image_size = image_height * image_width

# generator architecture ---> 0: resnet 9 blocks, 1: resnet 12 blocks, 2: unet
generator_type = 0
# discriminator architecture ---> 0: pixel level discriminator, 1: patch discriminator
discriminator_type = 1

# hyperparameters

initial_learning_rate = 0.0002
total_epochs_to_train = 300
max_images_to_train_per_epoch = 1000
learning_rate_drop_epoch = 150
batch_size = 1
image_pool_size = 1
num_filters_generator = 64
num_filters_discriminator = 64
cycle_loss_weight = 10.0
identity_loss_weight = 10.0
generator_loss_weight = 1.0
beta_1 = 0.5
beta_2 = 0.9
unet_dropout_rate = 0.5
patch_size = 128


# class for creating the cycleGAN model
class CycleGAN():

    # function for creating cycle GAN model graph in tensorflow
	def create_cycleGAN_model(self):

        # creates placeholders for original images of the two domains
		self.original_A = tf.placeholder(tf.float32, [batch_size, image_width, image_height, image_channels], name="original_A")
		self.original_B = tf.placeholder(tf.float32, [batch_size, image_width, image_height, image_channels], name="original_B")

		# creates placeholder for generated_image_temp images
		self.generated_pool_A = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name="generated_pool_A")
		self.generated_pool_B = tf.placeholder(tf.float32, [None, image_width, image_height, image_channels], name="generated_pool_B")




		# variable for keeping track of the number of epochs the model has trained
		self.global_step = tf.Variable(0, name="global_step", trainable=False)

		self.total_generated_inputs = 0

		self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")




		with tf.variable_scope("cycleGAN") as scope:

			# naming convention: 0-> original image; 1-> generated_image_temp image; 2-> cyclic image

			# generated_image_temp images in the other domain (1)

			if generator_type == 0:
				self.generated_image_B = create_generator_resnet_9_blocks(self.original_A, num_filters_generator, name="generator_A_to_B")
				self.generated_image_A = create_generator_resnet_9_blocks(self.original_B, num_filters_generator, name="generator_B_to_A")

			elif generator_type == 1:
				self.generated_image_B = create_generator_resnet_12_blocks(self.original_A, num_filters_generator, name="generator_A_to_B")
				self.generated_image_A = create_generator_resnet_12_blocks(self.original_B, num_filters_generator, name="generator_B_to_A")
			elif generator_type == 2:
				self.generated_image_B = create_generator_unet(self.original_A, num_filters_generator, unet_dropout_rate, name="generator_A_to_B")
				self.generated_image_A = create_generator_unet(self.original_B, num_filters_generator, unet_dropout_rate, name="generator_B_to_A")


			# build corresponding discriminators (0)
			if discriminator_type == 0:
				self.classification_original_A = create_discriminator_pixel_level(self.original_A, num_filters_discriminator, name="discriminator_A")
				self.classification_original_B = create_discriminator_pixel_level(self.original_B, num_filters_discriminator, name="discriminator_B")
			elif discriminator_type == 1:
				self.classification_original_A = create_discriminator_patch(self.original_A, num_filters_discriminator, patch_size, name="discriminator_A")
				self.classification_original_B = create_discriminator_patch(self.original_B, num_filters_discriminator, patch_size, name="discriminator_B")

			# reuse the variables from previous training
			scope.reuse_variables()

			# build corresponding discriminators (1)
			if discriminator_type == 0:
				self.classification_generated_B = create_discriminator_pixel_level(self.generated_image_B, num_filters_discriminator, name="discriminator_B")
				self.classification_generated_A = create_discriminator_pixel_level(self.generated_image_A, num_filters_discriminator, name="discriminator_A")
			elif discriminator_type == 1:
				self.classification_generated_B = create_discriminator_patch(self.generated_image_B, num_filters_discriminator, patch_size, name="discriminator_B")
				self.classification_generated_A = create_discriminator_patch(self.generated_image_A, num_filters_discriminator, patch_size, name="discriminator_A")

			# generator for cyclic images (2)
			if generator_type == 0:
				self.cyclic_A = create_generator_resnet_9_blocks(self.generated_image_B, num_filters_generator, name="generator_B_to_A")
				self.cyclic_B = create_generator_resnet_9_blocks(self.generated_image_A, num_filters_generator,  name="generator_A_to_B")
			elif generator_type == 1:
				self.cyclic_A = create_generator_resnet_12_blocks(self.generated_image_B, num_filters_generator, name="generator_B_to_A")
				self.cyclic_B = create_generator_resnet_12_blocks(self.generated_image_A, num_filters_generator, name="generator_A_to_B")
			elif generator_type == 2:
				self.cyclic_A = create_generator_unet(self.generated_image_B, num_filters_generator, unet_dropout_rate, name="generator_B_to_A")
				self.cyclic_B = create_generator_unet(self.generated_image_A, num_filters_generator, unet_dropout_rate, name="generator_A_to_B")

			# reuse the variables
			scope.reuse_variables()

			# decision of generated_image_temp image pool
			if discriminator_type == 0:
				self.classification_generated_pool_A = create_discriminator_pixel_level(self.generated_pool_A, num_filters_discriminator, name="discriminator_A")
				self.classification_generated_pool_B = create_discriminator_pixel_level(self.generated_pool_B, num_filters_discriminator, name="discriminator_B")
			elif discriminator_type == 1:
				self.classification_generated_pool_A = create_discriminator_patch(self.generated_pool_A, num_filters_discriminator, patch_size, name="discriminator_A")
				self.classification_generated_pool_B = create_discriminator_patch(self.generated_pool_B, num_filters_discriminator, patch_size, name="discriminator_B")


    # function for building loss graph
	def create_loss_function(self):

		# calculate the cyclic loss
		# tries to minimize the difference between original image and cyclic image
		# cyclic loss is calculated by combining the loss in both directions
		cyclic_loss = tf.reduce_mean(tf.abs(self.original_A - self.cyclic_A)) \
		            + tf.reduce_mean(tf.abs(self.original_B - self.cyclic_B))

		identity_loss = tf.reduce_mean(tf.abs(self.generated_image_A - self.original_A)) \
		            + tf.reduce_mean(tf.abs(self.generated_image_B - self.original_B))

		# tries to make the value close to 1
		# classification_generated_A value of close to 1 is classified as real
		generator_partial_loss_A = tf.reduce_mean(tf.squared_difference(self.classification_generated_A, tf.ones_like(self.classification_generated_A)))
		generator_partial_loss_B = tf.reduce_mean(tf.squared_difference(self.classification_generated_B, tf.ones_like(self.classification_generated_B)))

		# calculation of generator loss, cycle_loss_weight is used to give more importance to the cyclic loss than the discriminator loss
		generator_loss_A = cyclic_loss * cycle_loss_weight + generator_partial_loss_A * generator_loss_factor + identity_loss * identity_loss_weight
		generator_loss_B = cyclic_loss * cycle_loss_weight + generator_partial_loss_B * generator_loss_factor + identity_loss * identity_loss_weweight		# tries to minimize (make close to 0) classification_generated_pool_* score so that they can be classified as generated_image_temp
		# tries to maximize (make close to 1) the classification_original_* score so that they can be classified as real
		# takes the averag of the two scores
		discriminator_loss_A = (tf.reduce_mean(tf.square(self.classification_generated_pool_A)) \
		                        + tf.reduce_mean(tf.squared_difference(self.classification_original_A, 1))) \
		                        / 2.0
		discriminator_loss_B = (tf.reduce_mean(tf.square(self.classification_generated_pool_B)) \
		                        + tf.reduce_mean(tf.squared_difference(self.classification_original_B, 1))) \
		                        / 2.0

		# uses Adam optimizer to update the weights
		# beta 1: the exponential decay rate for the 1st moment estimates
		optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=beta_1, beta2=beta_2)

		# gets the trainable variables of the model
		self.cycleGAN_variables = tf.trainable_variables()

		discriminator_A_variables = [var for var in self.cycleGAN_variables if 'discriminator_A' in var.name]
		generator_A_to_B_variables = [var for var in self.cycleGAN_variables if 'generator_A_to_B' in var.name]
		discriminator_B_variables = [var for var in self.cycleGAN_variables if 'discriminator_B' in var.name]
		generator_B_to_A_variables = [var for var in self.cycleGAN_variables if 'generator_B_to_A' in var.name]

		# passes the corresponding variables in the optimizer functions
		self.discriminator_A_trainer = optimizer.minimize(discriminator_loss_A, var_list=discriminator_A_variables)
		self.discriminator_B_trainer = optimizer.minimize(discriminator_loss_B, var_list=discriminator_B_variables)
		self.generator_A_to_B_trainer = optimizer.minimize(generator_loss_A, var_list=generator_A_to_B_variables)
		self.generator_B_to_A_trainer = optimizer.minimize(generator_loss_B, var_list=generator_B_to_A_variables)

		# prints all the variables in the model
		for var in self.cycleGAN_variables: 
			print(var.name)

        # losses to be displayed in tensorboard
		self.generator_A_to_B_loss_summary = tf.summary.scalar("generator_A_to_B_loss", generator_loss_A)
		self.generator_B_to_A_loss_summary = tf.summary.scalar("generator_B_to_A_loss", generator_loss_B)
		self.discriminator_A_loss_summary = tf.summary.scalar("discriminator_A_loss", discriminator_loss_A)
		self.discriminator_B_loss_summary = tf.summary.scalar("discriminator_B_loss", discriminator_loss_B)
		self.cyclic_loss_summary = tf.summary.scalar("cyclic_loss", cyclic_loss)
		self.identity_loss_summary = tf.summary.scalar("identity_loss", identity_loss)


    # function for training the clcyeGAN model
	def train(self):

		self.pool = ImagePool(image_pool_size)

		# loads the training images into memory
		self.import_datasets_train()  

		# create the cycle GAN model for training
		self.create_cycleGAN_model()

		#Loss function calculations
		self.create_loss_function()

		# opt for initializing all variables
		init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])

		# for saving and restoring variables
		saver = tf.train.Saver()     

		with tf.Session() as sess:

			# initialize all variables
			sess.run(init)

			# read the input images and convert them to nd arrays
			self.input_read_train(sess)

			# resumes training from the last checkpoint if restore_condition is set to true
			if restore_condition:
				checkpoint_filename = tf.train.latest_checkpoint(checkpoint_path)

				# manually supply checkpoint file
				#checkpoint_filename = tf.train.latest_checkpoint('./output/maps/{}/checkpoints/'.format(restore_folder))

				# restore the variables from the checkpoint
				saver.restore(sess, checkpoint_filename)

            # setup summary writer
			writer = tf.summary.FileWriter("./{}/summary".format(output_path))

            # create checkpoint path if it doesnot exist
			if not os.path.exists(checkpoint_path):
				os.makedirs(checkpoint_path)


			print('\n\n\nStarting to train model:\n\n\n')

			# Training Loop
			for current_epoch in range(sess.run(self.global_step), total_epochs_to_train):                

                # prints the current epoch to the console
				print ("\n-----------------------------------------------------------> Epoch: {}\n".format(current_epoch + 1))
				saver.save(sess, os.path.join(checkpoint_path, "cyclegan"), global_step=current_epoch)

				# specify the learning rate here
				current_learning_rate = initial_learning_rate

				# drop the learning rate if training has been done for learning_rate_drop_epoch number of epochs
				if (current_epoch > learning_rate_drop_epoch):
					current_learning_rate = current_learning_rate - current_learning_rate * (current_epoch - learning_rate_drop_epoch) / learning_rate_drop_epoch

                # saves training images
				for current_iteration in range(0, max_images_to_train_per_epoch):

					print("Epoch: {} -- Iteration: {}".format(current_epoch + 1, current_iteration + 1))

					if (current_iteration % 100 == 0): 
						if (save_training_images_condition):
							self.save_training_images(sess, current_epoch, current_iteration)


                    # optimizes the generator_A_to_B network
					generator_A_to_B_dict = {self.original_A: self.A_input[current_iteration], 
                    						self.original_B: self.B_input[current_iteration], 
                    						self.learning_rate: current_learning_rate};

					_, generated_B_temp, summary_string = sess.run([self.generator_A_to_B_trainer, 
																	self.generated_image_B, 
																	self.generator_A_to_B_loss_summary], 
																	feed_dict = generator_A_to_B_dict)

					writer.add_summary(summary_string, current_epoch * max_images_to_train_per_epoch + current_iteration)                    
					#generated_B_temp_1 = self.generated_image_pool(self.total_generated_inputs, generated_B_temp, self.generated_images_B)


					# optimizes the generator_B_to_A network
					generator_B_to_A_dict = {self.original_A: self.A_input[current_iteration], 
											self.original_B: self.B_input[current_iteration], 
											self.learning_rate: current_learning_rate}

					_, generated_A_temp, summary_string = sess.run([self.generator_B_to_A_trainer, 
																	self.generated_image_A, 
																	self.generator_B_to_A_loss_summary], 
																	feed_dict = generator_B_to_A_dict)

					writer.add_summary(summary_string, current_epoch * max_images_to_train_per_epoch + current_iteration)
					#generated_A_temp_1 = self.generated_image_pool(self.total_generated_inputs, generated_A_temp, self.generated_images_A)

					[generated_A_temp_1, generated_B_temp_1] = self.pool([generated_A_temp, generated_B_temp])



					# optimizes the discriminator_B network
					discriminator_B_dict = {self.original_A: self.A_input[current_iteration], 
											self.original_B: self.B_input[current_iteration], 
											self.learning_rate: current_learning_rate, 
											self.generated_pool_B: generated_B_temp_1}

					_, summary_string = sess.run([self.discriminator_B_trainer, 
												self.discriminator_B_loss_summary], 
												feed_dict = discriminator_B_dict)

					writer.add_summary(summary_string, current_epoch * max_images_to_train_per_epoch + current_iteration)
                    
                    
                    # optimizes the discriminator_A network
					discriminator_A_dict = {self.original_A: self.A_input[current_iteration], 
											self.original_B: self.B_input[current_iteration], 
											self.learning_rate: current_learning_rate, 
											self.generated_pool_A: generated_A_temp_1}

					_, summary_string, cyclic_loss_string, identity_loss_string = sess.run([self.discriminator_A_trainer, 
																							self.discriminator_A_loss_summary, 
																							self.cyclic_loss_summary,
																							self.identity_loss_summary],
																							feed_dict = discriminator_A_dict)

					writer.add_summary(summary_string, current_epoch * max_images_to_train_per_epoch + current_iteration)


					# saves the summary of the cyclic loss
					#cyclic_loss_string = sess.run([self.cyclic_loss_summary], feed_dict)
					writer.add_summary(cyclic_loss_string, current_epoch * max_images_to_train_per_epoch + current_iteration)
					writer.add_summary(identity_loss_string, current_epoch * max_images_to_train_per_epoch + current_iteration)

					self.total_generated_inputs += 1
            	
            	# increases the global epoch by 1
				sess.run(tf.assign(self.global_step, current_epoch + 1))

			# saves the graph
			writer.add_graph(sess.graph)


            
	# function for testing the model
	def test(self):

		# import the dataset
		self.import_datasets_test()

		# create the cycleGAN tensorflow model
		self.create_cycleGAN_model()

		# create saver function and initialize tensorflow variables 
		saver = tf.train.Saver()
		init = ([tf.global_variables_initializer(), 
				tf.local_variables_initializer()])

		# initialize the tensorflow session
		with tf.Session() as sess:

			sess.run(init)
			self.input_read_test(sess)

			# load the pre-trained weights and biases from the latest checkpoint
			checkpoint_filename = tf.train.latest_checkpoint(checkpoint_path)
			saver.restore(sess, checkpoint_filename)

			# create the folder for saving images if it does not exist
			if not os.path.exists(test_path):
				os.makedirs(test_path)

			a_to_b_path = test_path + "/A_to_B"
			if not os.path.exists(a_to_b_path):
				os.makedirs(a_to_b_path)  
				
			index_path = os.path.join(a_to_b_path, "A_to_B_index.html")
			index = open(index_path, "w")

			index.write("<html><body><table><tr>")
			index.write("<th>#</th>")
			index.write("<th>original A</th>")
			index.write("<th>generated B</th>")
			index.write("<th>Cyclic A</th></tr>")

			print('\n\n\nStarting to test model ({}):\n\n\n'.format(dataset_name))

			for i in range(0, total_test_images):

				# run and obtain the generated and cyclic images
				generated_B_temp, cyclic_A = sess.run([self.generated_image_B, self.cyclic_A], feed_dict = {self.original_A: self.A_input[i]})

				print('Saving image - {}'.format(i + 1))
				# save the images
				if test_A_to_B:
					# A to B direction
					original_A_filename = "original_A_{}.jpg".format(i + 1)
					generated_B_filename = "generated_B_{}.jpg".format(i + 1)
					cyclic_A_filename = "cyclic_A_{}.jpg".format(i + 1)

					original_A_path = "{}/A_to_B/{}".format(test_path, original_A_filename)
					generated_B_path = "{}/A_to_B/{}".format(test_path, generated_B_filename)
					cyclic_A_path = "{}/A_to_B/{}".format(test_path, cyclic_A_filename)

					imsave(original_A_path, ((self.A_input[i][0] + 1) * 127.5).astype(np.uint8))
					imsave(generated_B_path, ((generated_B_temp[0] + 1) * 127.5).astype(np.uint8))
					imsave(cyclic_A_path, ((cyclic_A[0] + 1) * 127.5).astype(np.uint8))


					index.write('<td>{}</td>'.format(i+1))
					index.write("<td><img src='{}'></td>".format(original_A_filename))
					index.write("<td><img src='{}'></td>".format(generated_B_filename))
					index.write("<td><img src='{}'></td>".format(cyclic_A_filename))
					index.write("</tr>")

				if test_B_to_A:
					# B to A direction
					original_B_filename = "original_B_{}.jpg".format(i + 1)
					generated_A_filename = "generated_A_{}.jpg".format(i + 1)
					cyclic_B_filename = "cyclic_B_{}.jpg".format(i + 1)

					original_B_path = "{}/B_to_A/{}".format(test_path, original_B_filename)
					generated_A_path = "{}/B_to_A/{}".format(test_path, generated_A_filename)
					cyclic_B_path = "{}/B_to_A/{}".format(test_path, cyclic_B_filename)

					imsave(original_B_path, ((self.B_input[i][0] + 1) * 127.5).astype(np.uint8))
					imsave(generated_B_path, ((generated_A_temp[0] + 1) * 127.5).astype(np.uint8))
					imsave(cyclic_A_path, ((cyclic_B[0] + 1) * 127.5).astype(np.uint8))


					index.write('<td>{}</td>'.format(i+1))
					index.write("<td><img src='{}'></td>".format(original_B_filename))
					index.write("<td><img src='{}'></td>".format(generated_A_filename))
					index.write("<td><img src='{}'></td>".format(cyclic_B_filename))
					index.write("</tr>")

				if i == total_test_images:
					print('\n\n\nFinished Testing\n\n\n')
					index.close()

					
				
	# def run_inception(self):

	# 	# import the dataset
	# 	self.import_datasets_inception()

	# 	# create the cycleGAN tensorflow model
	# 	self.create_cycleGAN_model()

	# 	# create saver function and initialize tensorflow variables 
	# 	saver = tf.train.Saver()
	# 	init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])

	# 	# initialize the tensorflow session
	# 	with tf.Session() as sess:

	# 		sess.run(init)
	# 		self.input_read_inception(sess)
	# 		f = get_fid(self.A_input[0], self.A_input[0])
	# 		# print(len(self.A_input[0].shape))
	# 		# mean, std = get_inception_score(a)

	# 		print(f)

			


	''' 
	###############################################################################################################################
									functions for importing and saving the datasets and images
	###############################################################################################################################
	'''


	# function for reading in the datasets (train)
	def import_datasets_train(self):

        # fetch all the filenames
		filenames_A = tf.train.match_filenames_once("./input/{}/trainA/*.jpg".format(dataset_name))  
		filenames_B = tf.train.match_filenames_once("./input/{}/trainB/*.jpg".format(dataset_name)) 

        # number of images in each dataset
		self.queue_length_A = tf.size(filenames_A)           
		self.queue_length_B = tf.size(filenames_B)

		# enqueue the names of the files for execution pipeline
		filename_queue_A = tf.train.string_input_producer(filenames_A)
		filename_queue_B = tf.train.string_input_producer(filenames_B)

		# WholeFileReader object for reading image files
		image_reader = tf.WholeFileReader()

		# takes an image filename from the queue, gets the key and value of the image
		# * key: A string scalar Tensor. * value: A string scalar Tensor.
		# images are stored as arrays/collections
		_, image_file_A = image_reader.read(filename_queue_A)
		_, image_file_B = image_reader.read(filename_queue_B)

		# deode jpeg images to a unit8 tensors -> resizes the images to specified dimentions
		# -> divides the images by 127.5 and subtracts 1 for normalization
		# values range from [-1, 1]
		self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [image_height, image_width]), 127.5), 1)
		self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [image_height, image_width]), 127.5), 1)

    # function for reading in the datasets (test)
	def import_datasets_test(self):

		# custom input_dataset
		# if load_custom_test_dataset == True:
		# 	dataset_name = custom_test_dataset
		# dataset_name = 'summer2winterCustom'

        # fetch all the filenames
		filenames_A = tf.train.match_filenames_once("./input/{}/testA/*.jpg".format(dataset_name))  
		#filenames_B = tf.train.match_filenames_once("./input/{}/testB/*.jpg".format(dataset_name))

		# number of images in each dataset
		self.queue_length_A = tf.size(filenames_A)           
		#self.queue_length_B = tf.size(filenames_B)

		# enqueue the names of the files for execution pipeline
		filename_queue_A = tf.train.string_input_producer(filenames_A)
		#filename_queue_B = tf.train.string_input_producer(filenames_B)

		# WholeFileReader object for reading image files
		image_reader = tf.WholeFileReader()

		# takes an image filename from the queue, gets the key and value of the image
		# * key: A string scalar Tensor. * value: A string scalar Tensor.
		# images are stored as arrays/collections
		_, image_file_A = image_reader.read(filename_queue_A)
		#_, image_file_B = image_reader.read(filename_queue_B)

		# deode jpeg images to a unit8 tensors -> resizes the images to specified dimentions
		# -> divides the images by 127.5 and subtracts 1 for normalization
		# values range from [-1, 1]
		self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [image_height, image_width]), 127.5), 1)
		#self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [image_height, image_width]), 127.5), 1)


	# def import_datasets_inception(self):

	# 	# custom input_dataset
	# 	# if load_custom_test_dataset == True:
	# 	# 	dataset_name = custom_test_dataset
	# 	# dataset_name = 'summer2winterCustom'

 #        # fetch all the filenames
	# 	filenames_A = tf.train.match_filenames_once("inception/*.jpg")  
	# 	#filenames_B = tf.train.match_filenames_once("./input/{}/testB/*.jpg".format(dataset_name))

	# 	# number of images in each dataset
	# 	self.queue_length_A = tf.size(filenames_A)           
	# 	#self.queue_length_B = tf.size(filenames_B)

	# 	# enqueue the names of the files for execution pipeline
	# 	filename_queue_A = tf.train.string_input_producer(filenames_A)
	# 	#filename_queue_B = tf.train.string_input_producer(filenames_B)

	# 	# WholeFileReader object for reading image files
	# 	image_reader = tf.WholeFileReader()

	# 	# takes an image filename from the queue, gets the key and value of the image
	# 	# * key: A string scalar Tensor. * value: A string scalar Tensor.
	# 	# images are stored as arrays/collections
	# 	_, image_file_A = image_reader.read(filename_queue_A)
	# 	#_, image_file_B = image_reader.read(filename_queue_B)

	# 	# deode jpeg images to a unit8 tensors -> resizes the images to specified dimentions
	# 	# -> divides the images by 127.5 and subtracts 1 for normalization
	# 	# values range from [-1, 1]
	# 	self.image_A = tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [image_height, image_width])
	# 	#self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [image_height, image_width]), 127.5), 1)



	# function for loading images and convert into nd tensors
	def input_read_train(self, sess):

        # create coordinator for threads
		coord = tf.train.Coordinator()
		# get the list of threads
		threads = tf.train.start_queue_runners(coord=coord)

		# number of images in each domain
		num_files_A = sess.run(self.queue_length_A)
		num_files_B = sess.run(self.queue_length_B)

		# set all the pixels of  generated_images to 0
		self.generated_images_A = np.zeros((image_pool_size, 1, image_height, image_width, image_channels))
		self.generated_images_B = np.zeros((image_pool_size, 1, image_height, image_width, image_channels))


		self.A_input = np.zeros((max_images_to_train_per_epoch, batch_size, image_height, image_width, image_channels))
		self.B_input = np.zeros((max_images_to_train_per_epoch, batch_size, image_height, image_width, image_channels))

        # read in the images
		for i in range(max_images_to_train_per_epoch): 
			image_tensor = sess.run(self.image_A)

			if (image_tensor.size == image_size * batch_size * image_channels):
			    self.A_input[i] = image_tensor.reshape((batch_size, image_height, image_width, image_channels))

		for i in range(max_images_to_train_per_epoch):
			image_tensor = sess.run(self.image_B)

			if(image_tensor.size == image_size * batch_size * image_channels):
			    self.B_input[i] = image_tensor.reshape((batch_size, image_height, image_width, image_channels))


        # terminate the threads
		coord.request_stop()

		# handles exception for threads which are still running
		coord.join(threads)

	def input_read_test(self, sess):

        # create coordinator for threads
		coord = tf.train.Coordinator()
		# get the list of threads
		threads = tf.train.start_queue_runners(coord=coord)

		# number of images in each domain
		num_files_A = sess.run(self.queue_length_A)

		# set all the pixels of  generated_images to 0
		self.generated_images_B = np.zeros((image_pool_size, 1, image_height, image_width, image_channels))


		self.A_input = np.zeros((max_images_to_train_per_epoch, batch_size, image_height, image_width, image_channels))

        # read in the images
		for i in range(max_images_to_train_per_epoch): 
			image_tensor = sess.run(self.image_A)

			if (image_tensor.size == image_size * batch_size * image_channels):
			    self.A_input[i] = image_tensor.reshape((batch_size, image_height, image_width, image_channels))



        # terminate the threads
		coord.request_stop()

		# handles exception for threads which are still running
		coord.join(threads)


	# def input_read_inception(self, sess):

 #        # create coordinator for threads
	# 	coord = tf.train.Coordinator()
	# 	# get the list of threads
	# 	threads = tf.train.start_queue_runners(coord=coord)

	# 	# number of images in each domain
	# 	num_files_A = sess.run(self.queue_length_A)

	# 	# set all the pixels of  generated_images to 0
	# 	self.generated_images_B = np.zeros((image_pool_size, 1, image_channels, image_height, image_width))


	# 	self.A_input = np.zeros((max_images_to_train_per_epoch, batch_size,  image_channels, image_height, image_width))

 #        # read in the images
	# 	for i in range(max_images_to_train_per_epoch): 
	# 		image_tensor = sess.run(self.image_A)

	# 		if (image_tensor.size == image_size * batch_size * image_channels):
	# 		    self.A_input[i] = image_tensor.reshape((batch_size, image_channels, image_height, image_width))



 #        # terminate the threads
	# 	coord.request_stop()

	# 	# handles exception for threads which are still running
	# 	coord.join(threads)




	# function for saving training images
	def save_training_images(self, sess, current_epoch, current_iteration):
		# create output directory if it does not exists
		if not os.path.exists(images_path):
			os.makedirs(images_path)
			os.makedirs("{}/A_to_B".format(images_path))
			os.makedirs("{}/B_to_A".format(images_path))

		for i in range(0, num_train_images_to_save):
			temp_dict = {self.original_A: self.A_input[i], 
						self.original_B: self.B_input[i]}

			generated_A_temp, generated_B_temp, cyclic_A_temp, cyclic_B_temp = sess.run([self.generated_image_A, 
																						self.generated_image_B, 
																						self.cyclic_A, self.cyclic_B], 
																						feed_dict = temp_dict)

			# save the original images only for the first current_epoch
			# save A images
			imsave("{}/A_to_B/original_A_{}.jpg".format(images_path, i + 1), ((self.A_input[i][0] + 1) * 127.5).astype(np.uint8))

			# save B images
			imsave("{}/B_to_A/original_B_{}.jpg".format(images_path, i + 1), ((self.B_input[i][0] + 1) * 127.5).astype(np.uint8))

			# save the A -> G(A) images
			imsave("{}/A_to_B/generated_B_epoch_{}_iteration_{}_image_{}.jpg".format(images_path, current_epoch + 1, current_iteration + 1, i + 1), 
					((generated_B_temp[0] + 1) * 127.5).astype(np.uint8))
			# save the G(A) -> F(G(A)) images
			imsave("{}/A_to_B/cyclic_A_epoch_{}_iteration_{}_image_{}.jpg".format(images_path, current_epoch + 1, current_iteration + 1, i + 1), 
					((cyclic_A_temp[0]+1) * 127.5).astype(np.uint8))


			# save the B -> F(B) images
			imsave("{}/B_to_A/generated_A_epoch_{}_iteration_{}_image_{}.jpg".format(images_path, current_epoch + 1, current_iteration + 1, i + 1), 
					((generated_A_temp[0] + 1) * 127.5).astype(np.uint8))

			# save the F(B) -> G(F(B)) images
			imsave("{}/B_to_A/cyclic_B_epoch_{}_iteration_{}_image_{}.jpg".format(images_path, current_epoch + 1, current_iteration + 1, i + 1), 
					((cyclic_B_temp[0] + 1) * 127.5).astype(np.uint8))







def main():
    # creates an object of CycleGAN class
	cycleGAN = CycleGAN()

	#if training_condition is set to true -> starts training
	if training_condition:
		cycleGAN.train()
    # if test_condition is set to true -> starts testing
	elif test_condition:
		cycleGAN.test()
	# cycleGAN.run_inception()

if __name__ == '__main__':
	main()
