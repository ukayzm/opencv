# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to BING objectness saliency model")
ap.add_argument("-i", "--image", required=True,
                help="pathname to input image")
ap.add_argument("-n", "--max-detections", type=int, default=10,
                help="maximum # of detections to examine")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])

# initialize OpenCV's objectness saliency detector and set the path
# to the input model files
saliency = cv2.saliency.ObjectnessBING_create()
saliency.setTrainingPath(args["model"])

# compute the bounding box predictions used to indicate saliency
(success, saliencyMap) = saliency.conputeSaliency(image)
numDetections = saliencyMap.shape[0]

# loop over the detections
for i in range(0, min(numDetections, args["max_detections"])):
    # extract the bounding box coordinates
    (startX, startY, endX, endY) = saliencyMap[i].flatten()

    # randomly generate a color for the object and draw it on the image
    output = image.copy()
    color = np.random.randint(0, 255, size=(3,))
    color = [int(c) for c in color]
    cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)
