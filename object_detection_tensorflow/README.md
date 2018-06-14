# Object Detection and Instance Segmentation

Run TensorFlow object detection models on the video from the webcam in real time or still image file.

* camera.py - check webcam
* object_detector.py - run inference on the webcam frame
* image_detector.py - run inference on still image file
* live_streaming.py - send video to the network: http://IP_addr:5000/

All 4 files will be runnable like this:
```
$ python camera.py
$ python object_detector.py
$ python image_detector.py [-o OUTPUT_FILE] image_file
$ python live_streaming.py
```

You can specify the model to use when creating ObjectDetector object. Find the lines look like this in `object_detector.py`, `live_streaming.py` or `image_detector.py` and uncomment the model.

```python
     detector = ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
     #detector = ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28')
     #detector = ObjectDetector('pet', label_file='data/pet_label_map.pbtxt')
```

Refer to [Detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for the model name to use. The model is automatically downloaded if not exists in local.

Visit [https://ukayzm.github.io/tensorflow-instance-segmentation/](https://ukayzm.github.io/tensorflow-instance-segmentation/) or [https://ukayzm.github.io/pet-training/](https://ukayzm.github.io/pet-training/) for more information.
