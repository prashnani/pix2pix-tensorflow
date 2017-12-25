import tensorflow as tf
import numpy as np

def deconv(x, filter_height, filter_width, num_outputs, batch_size, stride_y, stride_x,  name, padding='SAME'):
	# deconv
	input_shape = x.get_shape()
	input_channels = input_shape[3]    
	output_channels = num_outputs
	output_height = input_shape[1]
	output_width = input_shape[2]
	
	weights = get_scope_variable(name, 'weights', shape=[filter_height, filter_width, output_channels, input_channels], initialvals=tf.random_normal_initializer(0, 0.02))      
	deconved = tf.nn.conv2d_transpose(x, weights, strides = [1, stride_y, stride_x, 1], padding = padding, output_shape = [int(batch_size),2*int(output_height),2*int(output_width),int(output_channels)])  
	return deconved
	
def lrelu(x,lrelu_alpha,name):
	x = tf.identity(x)
	return (0.5 * (1 + lrelu_alpha)) * x + (0.5 * (1 - lrelu_alpha)) * tf.abs(x)
	# return tf.maximum(x, tf.multiply(x, tf.constant(lrelu_alpha)), name = name)

def conv(x, filter_height, filter_width, num_outputs, stride_y, stride_x, name, padding='VALID', batchnorm = True, lrelu_alpha=0.2):
	# conv
	input_channels = int(x.get_shape()[3])
	weights = get_scope_variable(name, 'weights', shape=[filter_height, filter_width, input_channels, num_outputs], initialvals=tf.random_normal_initializer(0, 0.02))      
	conved = tf.nn.conv2d(x, weights, strides = [1, stride_y, stride_x, 1], padding = padding)
	return conved

def apply_batchnorm(x,name):
	x = tf.identity(x)
	channels = x.get_shape()[3]
	offset=  get_scope_variable(name, 'bn_offset', shape=[1], initialvals=tf.zeros_initializer)
	scale = get_scope_variable(name, 'bn_scale', shape=[1], initialvals=tf.random_normal_initializer(1.0, 0.02))
	# offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
	# scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
	mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
	variance_epsilon = 1e-5
	normalized = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=offset, scale=scale, variance_epsilon=variance_epsilon)
	return normalized

def dropout(x, dropout_rate):
	return tf.nn.dropout(x, dropout_rate)

def get_scope_variable(scope_name, var, shape=None, initialvals=None):
	with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
		v = tf.get_variable(var,shape,dtype=tf.float32, initializer=initialvals)
	return v

def get_text_file_lines(txt_file_name,shuffle=False):	
	txt_file = open(txt_file_name,'r')
	lines = txt_file.readlines()
	if shuffle == True:
		np.random.shuffle(lines)
	return lines