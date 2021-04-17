# load config
import json

with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    LOCAL_SERVER = config["LOCAL_SERVER"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

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
import time

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)


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


if __name__ == '__main__':
    # Main loop; infers sequentially until you press "q"
    while 1:
        # On "q" keypress, exit
        if (cv2.waitKey(1) == ord('q')):
            break

        # Capture start time to calculate fps
        start = time.time()

        # Synchronously get a prediction from the Roboflow Infer API
        image = infer()
        # And display the inference results
        cv2.imshow('image', image)

        # Print frames per second
        # print((1 / (time.time() - start)), " fps")

    # Release resources when finished
    video.release()
    cv2.destroyAllWindows()
