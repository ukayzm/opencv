# Face Clustering

Face clustering implementation using python.

* read video from file or web cam
* detect faces in the frame
* encode the faces
* save/load the encodings using pickle
* cluster the encodings using DBSCAN algorithm
* save the clustered faces in separated directory

```
$ python face_clustering.py -e video_file.mp4    # cluster faces in video_file
$ python face_clustering.py -e 0                 # video input from web cam
```

# Result Example

* ID1
<p align="center">
   <img src="jpg/ID1.montage.jpg">
</p>

* ID3
<p align="center">
   <img src="jpg/ID3.montage.jpg">
</p>

* ID6
<p align="center">
   <img src="jpg/ID6.montage.jpg">
</p>

* ID7
<p align="center">
   <img src="jpg/ID7.montage.jpg">
</p>
