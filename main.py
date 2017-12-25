import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import glob
from data_generator import *
from pix2pix_network import pix2pix_network
from random import shuffle
import skimage.io as io
import os
import argparse

curr_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", dest='num_epochs', type=int, default=200, help="specify number of epochs")
parser.add_argument("--lr", dest='lr', type=float, default=0.0002, help="specify learning rate")
parser.add_argument("--dropout", dest='dropout_rate', type=float, default=0.5, help="specify dropout")
parser.add_argument("--batch_size", dest='batch_size', type=float, default=1, help="specify batch size")
parser.add_argument("--dataset", dest='dataset_name', type=str, default='facades', help="specify dataset name")
parser.add_argument("--train_image_path", dest='train_image_path', type=str, help="specify path to training images")
parser.add_argument("--test_image_path", dest='test_image_path', type=str, help="specify path to test images")
parser.add_argument("--crop_size", dest='input_size', type=int, default=256, help="specify crop size of the final jittered input (sec.6.2 of the pix2pix paper https://arxiv.org/pdf/1611.07004.pdf)")
parser.add_argument("--enlarge_size", dest='enlarge_size', type=int, default=286, help="specify enlargement size from which to generate jittered input (sec.6.2 of the pix2pix paper https://arxiv.org/pdf/1611.07004.pdf)")
parser.add_argument("--out_dir", dest='out_dir', type=str, default=curr_path+'/test_outputs/', help="specify path to training images")
parser.add_argument("--checkpoint_name", dest='checkpoint_name', type=str, help="specify the checkpoint")
parser.add_argument("--mode", dest='mode', type=str, help="specify the checkpoint")

args = parser.parse_args()


def test(args):
	######## data IO
	out_dir = args.out_dir	
	test_image_names = glob.glob(args.test_image_path+'/*.jpg')	
	if not os.path.isdir(out_dir): os.mkdir(out_dir)
	# TF placeholder for graph input
	image_A = tf.placeholder(tf.float32, [None, args.input_size, args.input_size, 3])
	image_B = tf.placeholder(tf.float32, [None, args.input_size, args.input_size, 3])
	keep_prob = tf.placeholder(tf.float32)
	# Initialize model
	model = pix2pix_network(image_A,image_B,args.batch_size,keep_prob, weights_path='')
	# Loss
	D_loss, G_loss, G_loss_L1, G_loss_GAN = model.compute_loss()
	# Initialize a saver
	saver = tf.train.Saver(max_to_keep=None)
	# Config
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	######### Start training
	with tf.Session(config=config) as sess: 
		with tf.device('/gpu:0'):
			# Initialize all variables and start queue runners	
			sess.run(tf.local_variables_initializer())
			sess.run(tf.global_variables_initializer())
			# To continue training from one of the checkpoints
			if not args.checkpoint_name:
				raise IOError('In test mode, a checkpoint is expected.')
			saver.restore(sess, args.checkpoint_name)
			# Test network
			print 'generating network output'
			for curr_test_image_name in test_image_names:			
				splits = curr_test_image_name.split('/')
				splits = splits[len(splits)-1].split('.')
				print curr_test_image_name
				batch_A,batch_B = load_images_paired(list([curr_test_image_name]),
					is_train = False, true_size = args.input_size, enlarge_size = args.enlarge_size)
				fake_B = sess.run(model.generator_output(image_A), 
					feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), keep_prob: 1-args.dropout_rate})				
				io.imsave(out_dir+splits[0]+'_test_output_fakeB.png',(fake_B[0]+1.0)/2.0)
				io.imsave(out_dir+splits[0]+'_realB.png',(batch_B[0]+1.0)/2.0)
				io.imsave(out_dir+splits[0]+'_realA.png',(batch_A[0]+1.0)/2.0)

def train(args):
	######## data IO
	dataset_name = args.dataset_name
	image_names = glob.glob(args.train_image_path+'/*.jpg')
	shuffle(image_names)	
	test_image_names = glob.glob(args.test_image_path+'/*.jpg')

	######## Training variables
	num_epochs = args.num_epochs
	lr = args.lr	
	batch_size = args.batch_size
	total_train_images = len(image_names)
	num_iters_per_epoch = total_train_images/args.batch_size
	input_h = args.input_size
	input_w = args.input_size	
	dataset_name = args.dataset_name

	######### Prep for training
	# Path for tf.summary.FileWriter and to store model checkpoints
	filewriter_path = curr_path+'/'+dataset_name+"_pix2pix_training_info2/TBoard_files"
	checkpoint_path = curr_path+'/'+dataset_name+"_pix2pix_training_info2/"
	out_dir = checkpoint_path+'sample_outputs/'
	if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
	if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
	if not os.path.isdir(out_dir): os.mkdir(out_dir)

	# TF placeholder for graph input
	image_A = tf.placeholder(tf.float32, [None, input_h, input_w, 3])
	image_B = tf.placeholder(tf.float32, [None, input_h, input_w, 3])
	keep_prob = tf.placeholder(tf.float32)

	# Initialize model
	model = pix2pix_network(image_A,image_B,batch_size,keep_prob, weights_path='')

	# Loss
	D_loss, G_loss, G_loss_L1, G_loss_GAN = model.compute_loss()

	# Summary
	tf.summary.scalar("D_loss", D_loss)
	tf.summary.scalar("G_loss_GAN", G_loss_GAN)
	tf.summary.scalar("G_loss_L1", G_loss_L1)
	merged = tf.summary.merge_all()

	# Optimization
	D_vars = [v for v in tf.trainable_variables() if v.name.startswith("dis_")]
	D_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
	with tf.control_dependencies([D_train_op]):
		G_vars = [v for v in tf.trainable_variables() if v.name.startswith("gen_")]
		G_train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=G_vars)		

	# Initialize a saver and summary writer
	saver = tf.train.Saver(max_to_keep=None)

	# Config
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	######### Start training
	with tf.Session(config=config) as sess: 
		with tf.device('/gpu:0'):

			# Initialize all variables and start queue runners	
			sess.run(tf.local_variables_initializer())
			sess.run(tf.global_variables_initializer())
			threads = tf.train.start_queue_runners(sess=sess)
			train_writer = tf.summary.FileWriter(filewriter_path + '/train', sess.graph)

			# To continue training from one of the checkpoints
			if args.checkpoint_name:
				saver.restore(sess, args.checkpoint_name)
			
			start_time = time.time()
			# Loop over number of epochs
			start_epoch = 0
			for epoch in range(start_epoch,num_epochs):

				print "{} epoch: {}".format(datetime.now(), epoch)
				
				step = 0
				# Loop over iterations of an epoch
				D_loss_accum = 0.0
				G_loss_L1_accum = 0.0
				G_loss_GAN_accum = 0.0

				# Test network
				print 'generating network output'
				curr_test_image_name = np.random.choice(test_image_names, 1)
				batch_A,batch_B = load_images_paired(curr_test_image_name,is_train = False, true_size = args.input_size, enlarge_size = args.enlarge_size)
				fake_B = sess.run(model.generator_output(image_A), feed_dict=
					{image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), keep_prob: 1-args.dropout_rate})
				print curr_test_image_name[0][:-4]
				splits = curr_test_image_name[0].split('/')
				splits = splits[len(splits)-1].split('.')
				io.imsave(out_dir+splits[0]+'_epoch_'+str(epoch)+'.png',(np.concatenate(((batch_A[0]+1.0)/2.0,(batch_B[0]+1.0)/2.0,(fake_B[0]+1.0)/2.0),axis = 1)))

				for iter in np.arange(0,len(image_names),batch_size):

					# Get a batch of images (paired)				
					curr_image_names = image_names[iter*batch_size:(iter+1)*batch_size]
					batch_A,batch_B = load_images_paired(curr_image_names,is_train = True, true_size = args.input_size, enlarge_size = args.enlarge_size)

					# One training iteration
					summary, _, D_loss_curr_iter, G_loss_L1_curr_iter, G_loss_GAN_curr_iter = sess.run([merged, G_train_op, D_loss, G_loss_L1, G_loss_GAN], feed_dict={image_A: batch_A.astype('float32'), image_B: batch_B.astype('float32'), keep_prob: 1-args.dropout_rate})
					# Record losses for display
					G_loss_L1_accum = G_loss_L1_accum + G_loss_L1_curr_iter
					G_loss_GAN_accum = G_loss_GAN_accum + G_loss_GAN_curr_iter
					D_loss_accum = D_loss_accum + D_loss_curr_iter
					train_writer.add_summary(summary, epoch*len(image_names)/batch_size + iter)
					step += 1			
				
				end_time = time.time()
				print 'elapsed time for epoch '+str(epoch)+' = '+str(end_time-start_time)
				epoch = epoch+1
				G_loss_L1_accum = G_loss_L1_accum/num_iters_per_epoch
				G_loss_GAN_accum = G_loss_GAN_accum/num_iters_per_epoch
				D_loss_accum = D_loss_accum/num_iters_per_epoch

				print "G loss L1: "+str(G_loss_L1_accum)
				print "G loss GAN: "+str(G_loss_GAN_accum)
				print "D loss: "+str(D_loss_accum)

				# Save the most recent model
				for f in glob.glob(checkpoint_path+"model_epoch"+str(epoch-1)+"*"):
					os.remove(f)
				checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch)+'.ckpt')
				save_path = saver.save(sess, checkpoint_name)		

			train_writer.close()

def main(args):
	if args.mode == 'train':
		train(args)
	if args.mode == 'test':
		test(args)
	else:
		raise 'mode input should be train or test.'

if __name__ == '__main__':
    main(args)