import cv2
import tensorflow as tf
import numpy as np
import requests as rq
import imutils
import mediapipe as mp

ipAddress: str = "http://192.168.4.1/send"# 
cameraURL:str = "http://192.168.1.102:8080/shot.jpg"

predictedClass = None
predictionCount = 0
frameHoldThreshold = 10  

numberModel = tf.keras.models.load_model("numberModel.keras")

numbers = [str(i) for i in range(1, 11)]
letters = [chr(i) for i in range(65, 91)] + ["nothing", " "]

def getPredClassID(prediction):
    return int(np.argmax(prediction))

def getPredClassString(prediction, classNames):
    return classNames[getPredClassID(prediction)]

def getPredClass(prediction,classNames):
    classID = np.argmax(prediction)
    classString = classNames[classID]
    return classString

def sendPrediction(prediction, classNames):
    print(f"Sent prediction: {classNames[prediction]}")
    # rq.get(ipAddress, params={"data": classString})


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,     
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

while True:
    try:
        img_resp = rq.get(cameraURL)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        frame = imutils.resize(frame, height=400)  # resize for speed
    except Exception as e:
        print("Error fetching frame:", e)
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    cropped = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)
            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)

            cropped = frame[y_min:y_max, x_min:x_max]

    cv2.imshow('Sign Language Translator', frame)

    if cropped is not None and cropped.size > 0:
        img = cv2.resize(cropped, (200, 200))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = numberModel.predict(img, verbose=0)

        newClassID = getPredClassID(pred)
        newClass = numbers[newClassID]

        if predictedClass is None or predictedClass != newClassID:
            predictedClass = newClassID
            predictionCount = 0
        else:
            predictionCount += 1

        if predictionCount >= frameHoldThreshold:
            sendPrediction(predictedClass, numbers)
            predictionCount = 0

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
