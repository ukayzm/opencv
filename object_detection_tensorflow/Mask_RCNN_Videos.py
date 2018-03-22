
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[11]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
print(tf.__version__)


# ## Env setup

# ## Object detection imports
# Here are the imports from the object detection module.

# In[12]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[13]:


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/priya/Documents/mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28' + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Load a (frozen) Tensorflow model into memory.

# In[14]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[15]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[16]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[17]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[18]:


def detect_videos(image_np, sess, detection_graph):
    
    with detection_graph.as_default():
        
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
        output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image_np, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=1)
                
    return image_np


# In[19]:


def process_image(image):  
    
    global counter
    
    if counter%1 ==0:
   
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_np = detect_videos(image, sess, detection_graph) 

    counter +=1 
    
    return image


# In[20]:


filename = 'videos_in/cars_ppl.mp4'
new_loc = 'videos_out/cars_ppl_out.mp4'

counter = 0

white_output = new_loc
clip1 = VideoFileClip(filename).subclip(60,68)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')


# In[21]:


filename = 'videos_in/cars_ppl2.mp4'
new_loc = 'videos_out/cars_ppl2_out.mp4'

counter = 0

white_output = new_loc
clip1 = VideoFileClip(filename).subclip(64,72)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')


# In[22]:


filename = 'videos_in/kid_soccer.mp4'
new_loc = 'videos_out/kid_soccer_out.mp4'

counter = 0

white_output = new_loc
clip1 = VideoFileClip(filename).subclip(18,23)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')


# In[23]:


filename = 'videos_in/cat_dog.mp4'
new_loc = 'videos_out/cat_dog_out.mp4'

counter = 0

white_output = new_loc
clip1 = VideoFileClip(filename).subclip(120,125)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')


# In[24]:


filename = 'videos_in/fruits.mp4'
new_loc = 'videos_out/banana_out.mp4'

counter = 0

white_output = new_loc
clip1 = VideoFileClip(filename).subclip(79,85)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')


# In[25]:


filename = 'videos_in/apple_picking.mp4'
new_loc = 'videos_out/apple_picking_out.mp4'

counter = 0

white_output = new_loc
clip1 = VideoFileClip(filename).subclip(41,44)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')


# In[26]:


filename = 'videos_in/horse.mp4'
new_loc = 'videos_out/horse_out.mp4'

counter = 0

white_output = new_loc
clip1 = VideoFileClip(filename).subclip(45,49)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!s
get_ipython().magic(u'time white_clip.write_videofile(white_output, audio=False)')

