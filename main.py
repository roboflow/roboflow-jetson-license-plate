# load config
import json

# Get config variables
from PIL import Image

with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    LOCAL_SERVER = config["LOCAL_SERVER"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Local Server Link
if not LOCAL_SERVER:
    upload_url = "".join([
        "https://infer.roboflow.com/" + ROBOFLOW_MODEL,
        "?access_token=" + ROBOFLOW_API_KEY,
        "&name=YOUR_IMAGE.jpg"
    ])
else:
    upload_url = "".join([
        "http://127.0.0.1:9001/" + ROBOFLOW_MODEL,
        "?access_token=" + ROBOFLOW_API_KEY,
        "&name=YOUR_IMAGE.jpg"
    ])

import cv2
import base64
import requests
import matplotlib.pyplot as plt
import keras_ocr
import numpy as np

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()


# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get predictions from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).json()['predictions']

    # Draw all predictions
    for prediction in resp:
        # Save License Plate as image
        if prediction['class'] == "license-plate":
            getLiscensePlate(img, prediction['x'], prediction['y'], prediction['width'], prediction['height'])

        writeOnStream(prediction['x'], prediction['y'], prediction['width'], prediction['height'],
                      prediction['class'],
                      img)

    return img


def writeOnStream(x, y, width, height, className, frame):
    # Draw a Rectangle around detected image
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
                  (255, 0, 0), 2)

    # Draw filled box for class name
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
                  (255, 0, 0), cv2.FILLED)

    # Set label font + draw Text
    font = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), font, 0.5, (255, 255, 255), 1)


def getLiscensePlate(frame, x, y, width, height):
    # Crop license plate
    crop_frame = frame[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    # Save license Plate
    cv2.imwrite("plate.jpg", crop_frame)
    # Pre Process Image
    preprocessImage("plate.jpg")
    # Read image for OCR
    images = [keras_ocr.tools.read("plate.jpg")]
    # Get Predictions
    prediction_groups = pipeline.recognize(images)
    # Print the predictions
    for predictions in prediction_groups:
        for prediction in predictions:
            print(prediction[0])


def preprocessImage(image):
    # Read Image
    img = cv2.imread(image)
    # Resize Image
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    # Change Color Format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Kernel to filter image
    kernel = np.ones((1, 1), np.uint8)
    # Dilate + Erode image using kernel
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)
    # Save + Return image
    cv2.imwrite('processed.jpg', img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


if __name__ == '__main__':
    # Main loop; infers sequentially until you press "q"
    while True:
        # On "q" keypress, exit
        if (cv2.waitKey(1) == ord('q')):
            break

        # Synchronously get a prediction from the Roboflow Infer API
        image = infer()
        # And display the inference results
        cv2.imshow('image', image)
    # Release resources when finished
    video.release()
    cv2.destroyAllWindows()
