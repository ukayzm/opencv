# Object Detection and Instance Segmentation

Run TensorFlow object detection models on the video from the webcam in real time.

* camera.py - check webcam
* object_detector.py - run inference on the webcam frame
* live_streaming.py - send video to the network: http://IP_addr:5000/

All 3 files will be runnable like this:
```
$ python camera.py
$ python object_detector.py
$ python live_streaming.py
```

You can specify the model to use when creating ObjectDetector object. Change the line of `object_detector.py`
```
174     model = 'ssd_mobilenet_v1_coco_2017_11_17'
175     #model = 'mask_rcnn_inception_v2_coco_2018_01_28'
```
or the line of `live_streaming.py`.
```
15    model = 'ssd_mobilenet_v1_coco_2017_11_17'
16    #model = 'mask_rcnn_inception_v2_coco_2018_01_28'
```

Refer to [Detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for the model name to use. The model is automatically downloaded if not exists in local.

Visit [https://ukayzm.github.io/tensorflow-instance-segmentation/](https://ukayzm.github.io/tensorflow-instance-segmentation/) for more information.
