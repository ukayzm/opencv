# live_streaming.py

from flask import Flask, render_template, Response
import object_detector
import camera
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(fr):
    detector = ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
    #detector = ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28')
    #detector = ObjectDetector('pet', label_file='data/pet_label_map.pbtxt')

    cam = camera.VideoCamera()

    while True:
        frame = cam.get_frame()
        frame = detector.detect_objects(frame)

        ret, jpg = cv2.imencode('.jpg', frame)
        jpg_bytes = jpg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(None),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
