# Face Recognition

Display webcam video in real time with person's names on the video.

* camera.py - check webcam
* face_recog.py - recognize faces on webcam frame
* live_streaming.py - send video over network http://IP_addr:5000/

All 3 files are runnable like this:
```
$ python camera.py
$ python face_recog.py
$ python live_streaming.py
```

Put picture with one person's face in `knowns` directory. Change the file name as the person's name like: `john.jpg` or `jane.jpg`. Then run `python face_recog.py`. Or `python live_streaming.py` to send video over network.

Visit [https://ukayzm.github.io/python-face-recognition/](https://ukayzm.github.io/python-face-recognition/) for more information.
