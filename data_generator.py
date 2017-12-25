import numpy as np
import tensorflow as tf
import skimage.io as io
import skimage.transform as transform
from random import randint
from random import shuffle

#images loaded in "paired" setting
def load_images_paired(img_names,is_train=True, true_size = 256, enlarge_size = 286):
  if is_train:
    resize_to = enlarge_size
  else:
    resize_to = true_size
  
  A_imgs = np.zeros((len(img_names),true_size,true_size,3)) # ASSUMING RGB FOR NOW
  B_imgs = np.zeros((len(img_names),true_size,true_size,3)) # ASSUMING RGB FOR NOW
  iter = 0
  for name in img_names:
    paired_im = io.imread(name)
    # print name
    B = transform.resize(paired_im[:,0:true_size,:],[resize_to,resize_to,3])*2.0-1.0
    A = transform.resize(paired_im[:,true_size:true_size*2,:],[resize_to,resize_to,3])*2.0-1.0
    tl_h = randint(0,resize_to-true_size)
    tl_w = randint(0,resize_to-true_size)
    flipflag = randint(0,1)>0 and is_train
    A_imgs[iter,:,:,:] = flip_image(A[tl_h:tl_h+true_size,tl_w:tl_w+true_size,:],flipflag)
    B_imgs[iter,:,:,:] = flip_image(B[tl_h:tl_h+true_size,tl_w:tl_w+true_size,:],flipflag)
    # io.imsave('A.png',(A+1)/2)
    iter += 1  
  
  return A_imgs,B_imgs

def flip_image(img,flipflag):
  if flipflag:
    return np.fliplr(img)
  else:
    return img

#----------------- TF record Data IO ------------------
# using code from: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

def preprocess_images_tf(imgs):
    assert len(imgs.get_shape().as_list()) == 3
    # imgs_max = tf.reduce_max(imgs,axis=[0,1,2],keep_dims=True)

    imgs_normed = tf.subtract(tf.multiply(tf.divide(tf.to_float(imgs,name='ToFloat'),tf.constant(255.0)),tf.constant(2.0)),tf.constant(1.0))
    return imgs_normed

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord_data_paired(out_file_name, hazy_txt_file, clear_txt_file):

    writer = tf.python_io.TFRecordWriter(out_file_name)

    hazy_im_names = utils.get_text_file_lines(hazy_txt_file)
    clear_im_names = utils.get_text_file_lines(clear_txt_file)

    image_name_pairs = zip(hazy_im_names,clear_im_names)
    print image_name_pairs
    shuffle(image_name_pairs)

    for hazy_name, clear_name in image_name_pairs:
      print clear_name
      hazy_img = np.array(io.imread(hazy_name[:-1])) # [:-1] because \n needs to be removed from the name # convert RGB to BGR im1 = im1[:,:,::-1]
      hazy_img = np.array(np.multiply(transform.resize(hazy_img, [286,286]),255)).astype('uint8')
      clear_img = np.array(io.imread(clear_name[:-1]))
      clear_img = np.array(np.multiply(transform.resize(clear_img, [286,286]),255)).astype('uint8')
      print type(clear_img[0][0][0])
      print clear_img.shape
      h,w,c = hazy_img.shape

      hazy_str = hazy_img.tostring()
      clear_str = clear_img.tostring()
      print len(clear_str)
      # write the record
      record = tf.train.Example(features=tf.train.Features(feature={
      'h': _int64_feature(h),
      'w': _int64_feature(w),
      'hazy_str_raw': _bytes_feature(hazy_str),
      'clear_str_raw': _bytes_feature(clear_str)}))

      writer.write(record.SerializeToString())

    writer.close()

def decode_tfrecord_data_paired(file_name, batch_size_val, out_h, out_w):
    out_channel = 3

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name)
    features = tf.parse_single_example(
      serialized_example,      
      features={
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'hazy_str_raw': tf.FixedLenFeature([], tf.string),
        'clear_str_raw': tf.FixedLenFeature([], tf.string)
        })    

    img1_vec = tf.decode_raw(features['hazy_str_raw'], tf.uint8)
    img2_vec = tf.decode_raw(features['clear_str_raw'], tf.uint8)
    
    h = tf.cast(features['h'], tf.int32)
    w = tf.cast(features['w'], tf.int32)

    image_shape = tf.stack([h, w, out_channel])    

    img1 = tf.reshape(img1_vec, image_shape)
    print img1.get_shape().as_list()
    img1_normed = preprocess_images_tf(img1)

    img2 = tf.reshape(img2_vec, image_shape)    
    img2_normed = preprocess_images_tf(img2)
            
    off_h = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    off_w = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    tformed_img1 = tf.image.crop_to_bounding_box(image=img1_normed,
                                           offset_height = off_h, 
                                           offset_width = off_w, 
                                           target_height=out_h,
                                           target_width=out_w)
    
    tformed_img2 = tf.image.crop_to_bounding_box(image=img2_normed,
                                           offset_height = off_h,
                                           offset_width = off_w,
                                           target_height=out_h,
                                           target_width=out_w)
    
    
    images1, images2 = tf.train.shuffle_batch( [tformed_img1, tformed_img2],
                                                 batch_size=batch_size_val,
                                                 capacity=50, # can be changed to more appropriate values
                                                 num_threads=2,
                                                 min_after_dequeue=30)    
    return images1, images2

def write_tfrecord_data(out_file_name, image_names, image_dir): # NOTE! to be used only for unpaired GAN training

    writer = tf.python_io.TFRecordWriter(out_file_name)
    shuffle(image_names)
    for image_name in image_names:
        print image_dir+image_name[0:-1]
        img = np.array(io.imread(image_dir+image_name[0:-1])) # convert RGB to BGR im1 = im1[:,:,::-1]        
        img = transform.resize(img, [286,286])
        
        h,w,c = img.shape

        img_str = img.tostring()
        # write the record
        record = tf.train.Example(features=tf.train.Features(feature={
        'h': _int64_feature(h),
        'w': _int64_feature(w),
        'img_str_raw': _bytes_feature(img_str),
        'img_name': _bytes_feature(image_name)}))

        writer.write(record.SerializeToString())

    writer.close()

def get_features(file_name):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name)
    features = tf.parse_single_example(
      serialized_example,      
      features={
        'h': tf.FixedLenFeature([], tf.int64),
        'w': tf.FixedLenFeature([], tf.int64),
        'img_str_raw': tf.FixedLenFeature([], tf.string),
        'img_name': tf.FixedLenFeature([], tf.string)
        })    
    return features


def decode_tfrecord_data_two_inputs(file_name1, file_name2, batch_size_val, out_h, out_w, paired=True):
    out_channel = 3

    features1 = get_features(file_name1)
    features2 = get_features(file_name2)

    img1_vec = tf.decode_raw(features1['img_str_raw'], tf.uint8)
    img2_vec = tf.decode_raw(features2['img_str_raw'], tf.uint8)
    
    img1_name = tf.cast(features1['img_name'], tf.string)
    img2_name = tf.cast(features2['img_name'], tf.string)

    h = tf.cast(features1['h'], tf.int32)
    w = tf.cast(features1['w'], tf.int32)

    image_shape = tf.stack([h, w, out_channel])    

    img1 = tf.reshape(img1_vec, image_shape)
    img1_normed = preprocess_images_tf(img1)

    img2 = tf.reshape(img2_vec, image_shape)    
    img2_normed = preprocess_images_tf(img2)
            
    off_h = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    off_w = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    tformed_img1 = tf.image.crop_to_bounding_box(image=img1_normed,
                                           offset_height = off_h, 
                                           offset_width = off_w, 
                                           target_height=out_h,
                                           target_width=out_w)
    
    tformed_img2 = tf.image.crop_to_bounding_box(image=img2_normed,
                                           offset_height = off_h,
                                           offset_width = off_w,
                                           target_height=out_h,
                                           target_width=out_w)
    
    
    images1, images2 = tf.train.shuffle_batch( [tformed_img1, tformed_img2],
                                                 batch_size=batch_size_val,
                                                 capacity=50, # can be changed to more appropriate values
                                                 num_threads=2,
                                                 min_after_dequeue=50)    
    return images1, images2

def decode_tfrecord_data_single_input(input_file_name, batch_size_val, out_h, out_w):
    out_channel = 3

    # assuming two file names
    file_name = input_file_name

    reader = tf.TFRecordReader()

    features1 = get_features(file_name1)

    img1_vec = tf.decode_raw(features1['img_str_raw'], tf.uint8)
    
    img1_name = tf.cast(features1['img_name'], tf.string)

    h = tf.cast(features1['h'], tf.int32)
    w = tf.cast(features1['w'], tf.int32)
   
    image_shape = tf.pack([h, w, out_channel])    
    
    img1 = tf.reshape(img1_vec, image_shape)
    img1_normed = preprocess_images_tf(img1)

    off_h = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2
    off_w = randint(0,29) #286 - 256 : see appendix of pix2pix paper; section 6.2

    tformed_img1 = tf.image.crop_to_bounding_box(image=img1_normed,
                                           offset_height = off_h, #this can be changed to add "jitter" as per pix2pix paper
                                           offset_width = off_w, #this can be changed to add "jitter" as per pix2pix paper
                                           target_height=out_h,
                                           target_width=out_w)
        
    
    images1 = tf.train.shuffle_batch(tformed_img1,
                                                 batch_size=batch_size_val,
                                                 capacity=30, # can be changed to more appropriate values
                                                 num_threads=2,
                                                 min_after_dequeue=10)    
    return images1