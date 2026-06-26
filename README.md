# ASL Real-Time Translator

A robust Sign Language Translation system that utilizes Computer Vision to interpret hand gestures in real-time. This project explores two different implementation approaches: one using **MediaPipe** for landmark detection and another using **YOLO (You Only Look Once)** for deep learning-based object detection.

## Project Overview

This system captures video input (either from a local webcam or an IP camera), processes the frames to detect hand gestures, and translates them into corresponding signs or characters. It includes a frame-stability mechanism to ensure accurate, non-flickering predictions.

## Key Features

- **Dual-Model Support:** 
  - *MediaPipe implementation:* Efficient hand landmark detection for precise gesture tracking.
  - *YOLO implementation:* High-performance object detection model (YOLOv11) for real-time classification.
- **Frame-Stability Logic:** Implements a threshold-based counter (`frameHoldThreshold`) to prevent unstable predictions and ensure the gesture is sustained before triggering an action.
- **Hardware Integration:** Ready-to-use functions for sending translation data to external IoT devices via HTTP requests.
- **Flexible Input:** Supports both local laptop webcams and network-based (IP) cameras.

## Technical Stack

- **Computer Vision:** OpenCV (`cv2`)
- **Deep Learning:** TensorFlow/Keras & Ultralytics YOLO
- **Hand Tracking:** MediaPipe
- **Communication:** `requests` (for HTTP-based IoT control)
- **Image Processing:** `imutils` & `numpy`

## Project Structure

```text
├── best.pt              # YOLO model weights
├── numberModel.keras    # TensorFlow/Keras model
├── main.py              # Combined or main execution script
├── images/              # Project documentation and demo assets
└── README.md            # Project documentation
