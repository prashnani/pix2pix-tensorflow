import numpy as np
import tensorflow as tf 
from utils import *

class pix2pix_network(object):

	def __init__(self, image_A_batch, image_B_batch, batch_size, dropout_rate, weights_path=''):    
		# Parse input arguments into class variables
		self.image_A = image_A_batch
		self.image_B = image_B_batch
		self.batch_size = batch_size
		self.dropout_rate = dropout_rate  
		self.WEIGHTS_PATH = weights_path
		self.l1_Weight = 100.0

	def generator_output(self,image_A_input):		

		# NOTE! the order of operations (as per aauthor's original code - https://github.com/phillipi/pix2pix) is:
		# non-linearity (if needed) -> conv -> batchnorm (if needed) for each layer
		scope_name = 'gen_e1'
		self.gen_e1 = apply_batchnorm(conv(image_A_input, 4, 4, 64, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		scope_name = 'gen_e2'
		self.gen_e2 = apply_batchnorm(conv(lrelu(self.gen_e1,lrelu_alpha=0.2,name = scope_name), 4, 4, 128, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		scope_name = 'gen_e3'
		self.gen_e3 = apply_batchnorm(conv(lrelu(self.gen_e2,lrelu_alpha=0.2,name = scope_name), 4, 4, 256, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		scope_name = 'gen_e4'
		self.gen_e4 = apply_batchnorm(conv(lrelu(self.gen_e3,lrelu_alpha=0.2,name = scope_name), 4, 4, 512, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		scope_name = 'gen_e5'
		self.gen_e5 = apply_batchnorm(conv(lrelu(self.gen_e4,lrelu_alpha=0.2,name = scope_name), 4, 4, 512, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		scope_name = 'gen_e6'
		self.gen_e6 = apply_batchnorm(conv(lrelu(self.gen_e5,lrelu_alpha=0.2,name = scope_name), 4, 4, 512, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		scope_name = 'gen_e7'
		self.gen_e7 = apply_batchnorm(conv(lrelu(self.gen_e6,lrelu_alpha=0.2,name = scope_name), 4, 4, 512, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		scope_name = 'gen_e8'
		self.gen_e8 = conv(lrelu(self.gen_e7,lrelu_alpha=0.2,name = scope_name), 4, 4, 512, 2, 2, padding = 'SAME', name=scope_name)

		scope_name = 'gen_d1'
		self.gen_d1 = apply_batchnorm(deconv(lrelu(self.gen_e8,lrelu_alpha=0.0, name = scope_name), 4, 4, 512, self.batch_size, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		self.gen_d1_dropout = dropout(self.gen_d1, self.dropout_rate)
		scope_name = 'gen_d2'
		self.gen_d2 = apply_batchnorm(deconv(lrelu(tf.concat([self.gen_d1_dropout, self.gen_e7],3),lrelu_alpha=0.0, name = scope_name), 4, 4, 512, self.batch_size, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		self.gen_d2_dropout = dropout(self.gen_d2, self.dropout_rate)		
		scope_name = 'gen_d3'
		self.gen_d3 = apply_batchnorm(deconv(lrelu(tf.concat([self.gen_d2_dropout, self.gen_e6],3), lrelu_alpha=0.0, name = scope_name), 4, 4, 512, self.batch_size, 2, 2, padding = 'SAME', name = scope_name),name = scope_name)
		self.gen_d3_dropout = dropout(self.gen_d3, self.dropout_rate)				
		scope_name = 'gen_d4'
		self.gen_d4 = apply_batchnorm(deconv(lrelu(tf.concat([self.gen_d3_dropout, self.gen_e5],3), lrelu_alpha=0.0, name = scope_name), 4, 4, 512, self.batch_size, 2, 2, padding = 'SAME', name = scope_name), name = scope_name)
		scope_name = 'gen_d5'
		self.gen_d5 = apply_batchnorm(deconv(lrelu(tf.concat([self.gen_d4, self.gen_e4],3), lrelu_alpha=0.0, name = scope_name), 4, 4, 256, self.batch_size, 2, 2, padding = 'SAME', name = scope_name), name = scope_name)
		scope_name = 'gen_d6'
		self.gen_d6 = apply_batchnorm(deconv(lrelu(tf.concat([self.gen_d5, self.gen_e3],3), lrelu_alpha=0.0, name = scope_name), 4, 4, 128, self.batch_size, 2, 2, padding = 'SAME', name = scope_name), name = scope_name)
		scope_name = 'gen_d7'
		self.gen_d7 = apply_batchnorm(deconv(lrelu(tf.concat([self.gen_d6, self.gen_e2],3), lrelu_alpha=0.0, name = scope_name), 4, 4, 64, self.batch_size, 2, 2, padding = 'SAME', name = scope_name), name = scope_name)
		scope_name = 'gen_d8'
		self.gen_d8 = deconv(lrelu(tf.concat([self.gen_d7, self.gen_e1],3), lrelu_alpha=0.0, name = scope_name), 4, 4, 3, self.batch_size, 2, 2, padding = 'SAME', name = scope_name)
		# generated output
		self.fake_B = tf.nn.tanh(self.gen_d8)
		return self.fake_B

	def discriminator_output(self, B_input): # 70x70 discriminator
		discrim_input = tf.concat([self.image_A,B_input],3)
		scope_name = 'dis_conv1'
		self.dis_conv1 = lrelu(conv(discrim_input,4,4,64,2,2,padding='SAME',name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv2'
		self.dis_conv2 = lrelu(apply_batchnorm(conv(self.dis_conv1,4,4,128,2,2,padding='SAME',name=scope_name),name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv3'
		self.dis_conv3 = lrelu(apply_batchnorm(conv(self.dis_conv2,4,4,256,2,2,padding='SAME',name=scope_name),name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv4'
		self.dis_conv4 = lrelu(apply_batchnorm(conv(self.dis_conv3,4,4,512,1,1,padding='SAME',name=scope_name),name=scope_name),lrelu_alpha=0.2, name=scope_name)
		scope_name = 'dis_conv5'
		self.dis_conv5 = conv(self.dis_conv4,4,4,1,1,1,padding='SAME',name=scope_name)
		self.dis_out_per_patch = tf.reshape(self.dis_conv5,[self.batch_size,-1])
		return tf.sigmoid(self.dis_out_per_patch)

	def compute_loss(self):
		eps = 1e-12
		fake_B = self.generator_output(self.image_A)
		fake_output_D = self.discriminator_output(fake_B)
		real_output_D = self.discriminator_output(self.image_B)		
		self.d_loss = tf.reduce_mean(-(tf.log(real_output_D + eps) + tf.log(1 - fake_output_D + eps)))     
		self.g_loss_l1= self.l1_Weight*tf.reduce_mean(tf.abs(fake_B - self.image_B))
		self.g_loss_gan = tf.reduce_mean(-tf.log(fake_output_D + eps))
		return self.d_loss, self.g_loss_l1 + self.g_loss_gan, self.g_loss_l1, self.g_loss_gan

	def load_initial_weights(self, session):
		# Load the weights into memory: this approach is adopted rather than standard random initialization to allow the
		# flexibility to load weights from a numpy file or other files.
		if self.WEIGHTS_PATH:
			print 'loading initial weights from '+ self.WEIGHTS_PATH
			weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()    
		# else:
		# 	print 'loading random weights'
		# 	weights_dict = get_random_weight_dictionary('pix2pix_initial_weights')
		# Loop over all layer names stored in the weights dict
		for op_name in weights_dict:          
			print op_name
			with tf.variable_scope(op_name) as scope:  
				for sub_op_name in weights_dict[op_name]:
	  				data = weights_dict[op_name][sub_op_name]
					var = get_scope_variable(name, sub_op_name, shape=[data.shape[0], data.shape[1], data.shape[2], data.shape[3]])
					session.run(var.assign(data))

#####################################################################################

# def get_random_weight_dictionary(net_name):
# 	if net_name == 'pix2pix_initial_weights':
# 		random_weight_dict = {}   
# 		# generator: encoder
# 		random_weight_dict.update({'gen_e1':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,3,64))}})		
# 		random_weight_dict.update({'gen_e2':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,64,128)), 
# 			'bn_offset': np.zeros((1,)),'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})		
# 		random_weight_dict.update({'gen_e3':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,128,256)),
# 			'bn_offset': np.zeros((1,)),'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})		
# 		random_weight_dict.update({'gen_e4':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,256,512)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})		
# 		random_weight_dict.update({'gen_e5':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,512)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})				
# 		random_weight_dict.update({'gen_e6':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,512)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})		
# 		random_weight_dict.update({'gen_e7':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,512)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})		
# 		random_weight_dict.update({'gen_e8':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,512))}})
# 		# generator: decoder
# 		random_weight_dict.update({'gen_d1':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,512)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'gen_d2':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,1024)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'gen_d3':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,1024)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})		
# 		random_weight_dict.update({'gen_d4':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,1024)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'gen_d5':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,256,1024)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'gen_d6':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,128,512)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'gen_d7':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,64,256)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'gen_d8':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,3,128))}})
# 		# discriminator
# 		random_weight_dict.update({'dis_conv1':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,6,64))}})
# 		random_weight_dict.update({'dis_conv2':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,64,128)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'dis_conv3':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,128,256)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'dis_conv4':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,256,512)), 
# 			'bn_offset': np.zeros((1,)), 'bn_scale': np.random.normal(loc = 1.0, scale = 0.02,size=(1,))}})
# 		random_weight_dict.update({'dis_conv5':{'weights': np.random.normal(loc = 0.0, scale = 0.02,size=(4,4,512,1))}})

# 	return random_weight_dict
	
#	# else:
#	# 	raise NotImplementedError
#	# 