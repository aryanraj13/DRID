# This script detects green objects and uses their motion to control a Snake game.

import cv2
import imutils
import numpy as np
from collections import deque
import time
import pyautogui
from threading import Thread

# Threaded webcam video stream class
class WebcamVideoStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Define HSV range for green color
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

buffer = 20
flag = 0
pts = deque(maxlen=buffer)
counter = 0
(dX, dY) = (0, 0)
direction = ''
last_pressed = ''

time.sleep(2)

width, height = pyautogui.size()
vs = WebcamVideoStream().start()
pyautogui.click(int(width / 2), int(height / 2))

while True:
    frame = vs.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=600)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_converted_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_converted_frame, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
            pts.appendleft(center)

    for i in np.arange(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        if counter >= 10 and i == 1 and len(pts) >= 10 and pts[-10] is not None:
            dX = pts[-10][0] - pts[i][0]
            dY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ('', '')

            if np.abs(dX) > 50:
                dirX = 'West' if np.sign(dX) == 1 else 'East'
            if np.abs(dY) > 50:
                dirY = 'North' if np.sign(dY) == 1 else 'South'

            direction = dirX if dirX != '' else dirY
            cv2.putText(frame, direction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Control game using arrow keys based on direction
    if direction == 'East' and last_pressed != 'right':
        pyautogui.press('right')
        last_pressed = 'right'
        print("Right Pressed")
    elif direction == 'West' and last_pressed != 'left':
        pyautogui.press('left')
        last_pressed = 'left'
        print("Left Pressed")
    elif direction == 'North' and last_pressed != 'up':
        pyautogui.press('up')
        last_pressed = 'up'
        print("Up Pressed")
    elif direction == 'South' and last_pressed != 'down':
        pyautogui.press('down')
        last_pressed = 'down'
        print("Down Pressed")

    cv2.imshow('Game Control Window', frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    if flag == 0:
        pyautogui.click(int(width / 2), int(height / 2))
        flag = 1

    if key == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
