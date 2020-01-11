from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import asyncio
import websockets
import pickle
import pandas as pd
import base64
from PIL import Image
import io
import re
import web_socket_server
import io
import PIL
import json

model_face_detection_path = 'haarcascade_frontalface_default.xml'


face_detection = cv2.CascadeClassifier(model_face_detection_path)



EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",

            "neutral"]

async def detect(websocket, path):

    while True:
        message = await websocket.recv()
        obj = json.loads(message);
        if obj["net"] == 'vgg':
            emotion_model_path = 'vgg16_vgg16.76-0.63.hdf5'
        else:
            emotion_model_path = 'models_mini_XCEPTION.88-0.65.hdf5'

        emotion_classifier = load_model(emotion_model_path, compile=False)

        decoded = Image.open(io.BytesIO(base64.b64decode(obj['data'])))
        img = np.array(decoded)
        frame = imutils.resize(img, width=400)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_detected = face_detection.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE) #detected object returned as list of rectangles;

        result_graph = np.ones((400, 300, 3), dtype="uint8")*255


        frameCopy = frame.copy()

        if len(faces_detected) > 0:
            biggest_face = sorted(faces_detected, reverse=True,

                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0] #returns the biggest detected face

            (X, Y, width, height) = biggest_face

            region_oi = gray_img[Y:Y + height, X:X + width]

            region_oi = cv2.resize(region_oi, (48, 48))

            region_oi = region_oi.astype("float") / 255.0

            region_oi = img_to_array(region_oi) #takes Numpy array and reurns a Numpy 3D array

            region_oi = np.expand_dims(region_oi, axis=0) #expands region_oi by adding another axis

            predictions = emotion_classifier.predict(region_oi)[0] #returns class (emotions) values for the roi

            emotion_probability = np.max(predictions) #takes the most probable emotions

            winning_label = EMOTIONS[predictions.argmax()] #label of the most probable emotion on face

        for (i, (emotion, probability)) in enumerate(zip(EMOTIONS, predictions)): #iterate over tuples of emotions with their predicitions

            result_info = "{}: {:.2f}%".format(emotion, probability * 100) #print the float percentage (2 digits after point) and the probability

            w = int(probability * 300)

            cv2.rectangle(result_graph, (1, (i * 50) + 30),

                          (w, (i * 50) + 70), (198,122,55), -1) #print a rectangle over result_graph and fill it with the specified color, coordinates: top left to bottom right

            cv2.putText(result_graph, result_info.upper(), (2, (i * 50) + 56),

                        cv2.FONT_ITALIC, 0.7,

                        (0,0,0), 2)

            cv2.putText(frameCopy, winning_label, (X, Y - 10),

                        cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)

            cv2.rectangle(frameCopy, (X, Y), (X + width, Y + height),

                          (0, 0, 255), 2)

        result_graph = cv2.resize(result_graph, (300,300))
        frameCopy = cv2.resize(frameCopy, (400,300))
        concat = cv2.hconcat([frameCopy, result_graph])
        concat_png = cv2.imencode('.png',concat)[1]
        concat_array = bytearray(concat_png)
        await websocket.send(concat_array)


start_server = websockets.serve(detect, "127.0.0.1", 5678)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
