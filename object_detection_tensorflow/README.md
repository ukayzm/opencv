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

You can specify the model to use when creating ObjectDetector object. It downloads the model if not exists in local. Refer to [Detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for the model name.
