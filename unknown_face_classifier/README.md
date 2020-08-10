# Unknown Face Classifier

* read video from file or web cam
* detect faces in the frame
* encode the faces
* compare the encoding with the previously saved faces
* if the encoding is similar with any person, save the face to the person
* if the encoding is similar with any face of unknowns, create new person
* otherwise, save the face to unknown faces

```
$ python face_classifier.py -h
usage: face_classifier.py [-h] -f FILE [-t THRESHOLD] [-c CAPTURE] [-s STOP]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  video file to detect or '0' to detect from web cam
  -t THRESHOLD, --threshold THRESHOLD
                        threshold of the similarity
  -c CAPTURE, --capture CAPTURE
                        # of frame to capture per second
  -s STOP, --stop STOP  stop encoding after # seconds
```


# Result Example

Tested with this video in YouTube (La La Land, Someone in the Crowd)
[![SomeoneInTheCrowd.mp4](https://img.youtube.com/vi/A7RmBgq4tT4/0.jpg)](https://www.youtube.com/watch?v=A7RmBgq4tT4)

```
$ python face_classifier.py -f ~/Videos/SomeoneInTheCrowd.mp4
source /home/rostude/Videos/SomeoneInTheCrowd.mp4
1280x512, 29.970030 frame/sec
capture every 30 frame
similarity shreshold: 0.55
press ^C to stop detecting immediately
frame 8310 @ time 277.277 takes 0.324 seconds - 9 persons, 4 unknown faces
similarities of persons:
person1    [ 0.000 0.496 0.468 0.306 0.447 0.697 0.678 0.560 0.680 ]
person2    [ 0.496 0.000 0.500 0.519 0.516 0.766 0.705 0.281 0.798 ]
person3    [ 0.468 0.500 0.000 0.371 0.397 0.665 0.662 0.515 0.689 ]
person4    [ 0.306 0.519 0.371 0.000 0.338 0.598 0.630 0.552 0.580 ]
person5    [ 0.447 0.516 0.397 0.338 0.000 0.634 0.467 0.565 0.629 ]
person6    [ 0.697 0.766 0.665 0.598 0.634 0.000 0.775 0.790 0.417 ]
person7    [ 0.678 0.705 0.662 0.630 0.467 0.775 0.000 0.769 0.794 ]
person8    [ 0.560 0.281 0.515 0.552 0.565 0.790 0.769 0.000 0.814 ]
person9    [ 0.680 0.798 0.689 0.580 0.629 0.417 0.794 0.814 0.000 ]
saving pictures in the directory 'SomeoneInTheCrowd'
```

person1
<p align="center">
   <img src="jpg/montage.person1-00.jpg">
</p>

person2
<p align="center">
   <img src="jpg/montage.person2-00.jpg">
</p>

person3
<p align="center">
   <img src="jpg/montage.person3-00.jpg">
</p>

person4
<p align="center">
   <img src="jpg/montage.person4-00.jpg">
</p>

person5
<p align="center">
   <img src="jpg/montage.person5-00.jpg">
</p>

person6
<p align="center">
   <img src="jpg/montage.person6-00.jpg">
</p>

person7
<p align="center">
   <img src="jpg/montage.person7-00.jpg">
</p>

person8
<p align="center">
   <img src="jpg/montage.person8-00.jpg">
</p>

person9
<p align="center">
   <img src="jpg/montage.person9-00.jpg">
</p>

unknown_faces
<p align="center">
   <img src="jpg/montage.unknown_faces-00.jpg">
</p>

