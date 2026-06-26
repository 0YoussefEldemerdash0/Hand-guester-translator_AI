# import cv2
# import numpy as np
# import requests as rq
# from ultralytics import YOLO

# # ==== CONFIG ====
# ipAddress: str = "http://192.168.1.5/send" 
# cameraURL: str = "http://192.168.1.5:8080/shot.jpg" 
# frameHoldThreshold = 10 
# predictedClass = None
# predictionCount = 0


# model = YOLO(r"best.pt")


# def sendPrediction(classID, labels):
#     classString = labels[classID]
#     print(f"Sent prediction: {classString}")
#     # rq.get(ipAddress, params={"data": classString})

# while True:
#     try:
#         img_resp = rq.get(cameraURL, timeout=5)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

#         if frame is None:
#             print("⚠️ Empty frame received, skipping...")
#             continue

#         frame = cv2.flip(frame, 1)

#     except Exception as e:
#         print("Error fetching frame:", e)
#         continue

#     try:
#         results = model(frame, verbose=False)
#     except Exception as e:
#         print("⚠️ YOLO failed on this frame:", e)
#         continue

#     detected = False
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             detected = True
#             classID = int(box.cls[0])     
#             conf = float(box.conf[0])    
#             label = model.names[classID]

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} {conf:.2f}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, (0, 255, 0), 2)

#             if predictedClass is None or predictedClass != classID:
#                 predictedClass = classID
#                 predictionCount = 0
#             else:
#                 predictionCount += 1

#             if predictionCount >= frameHoldThreshold:
#                 sendPrediction(classID, model.names)
#                 predictionCount = 0

#     if not detected:
#         predictedClass = None
#         predictionCount = 0

#     cv2.imshow("Sign Language Translator (YOLO)", frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#         break

# cv2.destroyAllWindows()


# ==========================================================================

import cv2
import numpy as np
import requests as rq
from ultralytics import YOLO

# ==== CONFIG ====
frameHoldThreshold = 10 
predictedClass = None
predictionCount = 0

# model = YOLO(r"X:\temp\runs\detect\asl_yolo11\weights\best.pt")
model = YOLO(r"best.pt")

def sendPrediction(classID, labels):
    classString = labels[classID]
    print(f"Sent prediction: {classString}")
    # rq.get(ipAddress, params={"data": classString})

# فتح كاميرا اللاب توب
cap = cv2.VideoCapture(0) 

while True:
    try:
        # قراءة الإطار من كاميرا اللاب توب
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to read from webcam, skipping...")
            continue

        frame = cv2.flip(frame, 1)

    except Exception as e:
        print("Error fetching frame:", e)
        continue

    try:
        results = model(frame, verbose=False)
    except Exception as e:
        print("⚠️ YOLO failed on this frame:", e)
        continue

    detected = False
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detected = True
            classID = int(box.cls[0]) 
            conf = float(box.conf[0]) 
            label = model.names[classID]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 6)

            if predictedClass is None or predictedClass != classID:
                predictedClass = classID
                predictionCount = 0
            else:
                predictionCount += 1

            if predictionCount >= frameHoldThreshold:
                sendPrediction(classID, model.names)
                predictionCount = 0

    if not detected:
        predictedClass = None
        predictionCount = 0

    cv2.imshow("Sign Language Translator (YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
