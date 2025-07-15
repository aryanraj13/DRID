from handTracker import *
import cv2
import mediapipe as mp
import numpy as np
import random
from collections import deque

class ColorRect():
    def __init__(self, x, y, w, h, color, text='', alpha=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha
        
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        alpha = self.alpha
        bg_rec = img[self.y:self.y + self.h, self.x:self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = res

        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w/2 - text_size[0][0]/2), int(self.y + self.h/2 + text_size[0][1]/2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        return (self.x + self.w > x > self.x) and (self.y + self.h > y > self.y)

# === INIT ===
detector = HandTracker(detectionCon=1)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), np.uint8)
px, py = 0, 0
color = (255, 0, 0)
brushSize = 5
eraserSize = 20
smooth_points = deque(maxlen=5)

colorsBtn = ColorRect(200, 0, 100, 100, (120, 255, 0), 'Colors')
colors = [
    ColorRect(400, 0, 100, 100, (0, 0, 255)),
    ColorRect(500, 0, 100, 100, (255, 0, 0)),
    ColorRect(600, 0, 100, 100, (0, 255, 0)),
    ColorRect(700, 0, 100, 100, (0, 255, 255)),
    ColorRect(800, 0, 100, 100, (0, 0, 0), "Eraser")
]
clear = ColorRect(900, 0, 100, 100, (100, 100, 100), "Clear")

pens = [ColorRect(1100, 50+100*i, 100, 100, (50, 50, 50), str(s)) for i, s in enumerate(range(5, 25, 5))]

penBtn = ColorRect(1100, 0, 100, 50, color, 'Pen')
pointerBtn = ColorRect(940, 0, 100, 50, (200, 100, 0), 'Pointer')  # moved left
boardBtn = ColorRect(50, 0, 100, 100, (255, 255, 0), 'Board')
whiteBoard = ColorRect(50, 120, 1020, 580, (255, 255, 255), alpha=0.6)

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True
usePointer = True

# HSV for green pointer
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=20)

while True:
    if coolingCounter:
        coolingCounter -= 1

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 720))
    center = None

    if usePointer:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                if radius > 10:
                    cv2.circle(frame, center, 10, (0, 255, 255), 2)
                    pts.appendleft(center)

                    # === Drawing ===
                    if whiteBoard.isOver(center[0], center[1]) and not hideBoard:
                        if px == 0 and py == 0:
                            px, py = center
                        if color == (0, 0, 0):
                            cv2.line(canvas, (px, py), center, color, eraserSize)
                        else:
                            cv2.line(canvas, (px, py), center, color, brushSize)
                        px, py = center
                    else:
                        px, py = 0, 0

                    # === UI interaction with pointer ===
                    if not hidePenSizes:
                        for pen in pens:
                            if pen.isOver(*center): brushSize = int(pen.text)
                            pen.alpha = 0 if pen.isOver(*center) else 0.5

                    if not hideColors:
                        for cb in colors:
                            if cb.isOver(*center): color = cb.color
                            cb.alpha = 0 if cb.isOver(*center) else 0.5

                        if clear.isOver(*center):
                            clear.alpha = 0
                            canvas = np.zeros((720, 1280, 3), np.uint8)
                        else:
                            clear.alpha = 0.5

                    if colorsBtn.isOver(*center) and not coolingCounter:
                        coolingCounter = 10
                        hideColors = not hideColors
                        colorsBtn.text = 'Colors' if hideColors else 'Hide'

                    if penBtn.isOver(*center) and not coolingCounter:
                        coolingCounter = 10
                        hidePenSizes = not hidePenSizes
                        penBtn.text = 'Pen' if hidePenSizes else 'Hide'

                    if boardBtn.isOver(*center) and not coolingCounter:
                        coolingCounter = 10
                        hideBoard = not hideBoard
                        boardBtn.text = 'Board' if hideBoard else 'Hide'

                    if pointerBtn.isOver(*center) and not coolingCounter:
                        coolingCounter = 10
                        usePointer = not usePointer

    else:
        detector.findHands(frame)
        positions = detector.getPostion(frame, draw=False)
        upFingers = detector.getUpFingers(frame)
        if upFingers and len(positions) > 8:
            x, y = positions[8]

            if upFingers[1] and not upFingers[2]:  # Drawing
                if whiteBoard.isOver(x, y) and not hideBoard:
                    cv2.circle(frame, (x, y), brushSize, color, -1)
                    smooth_points.append((x, y))
                    if len(smooth_points) >= 2:
                        avg_x = int(np.mean([pt[0] for pt in smooth_points]))
                        avg_y = int(np.mean([pt[1] for pt in smooth_points]))
                        if px == 0 and py == 0:
                            px, py = avg_x, avg_y
                        if color == (0, 0, 0):
                            cv2.line(canvas, (px, py), (avg_x, avg_y), color, eraserSize)
                        else:
                            cv2.line(canvas, (px, py), (avg_x, avg_y), color, brushSize)
                        px, py = avg_x, avg_y
                else:
                    px, py = 0, 0
                    smooth_points.clear()

            elif upFingers[1]:  # Interaction
                px, py = 0, 0
                smooth_points.clear()

                if not hidePenSizes:
                    for pen in pens:
                        if pen.isOver(x, y): brushSize = int(pen.text)
                        pen.alpha = 0 if pen.isOver(x, y) else 0.5

                if not hideColors:
                    for cb in colors:
                        if cb.isOver(x, y): color = cb.color
                        cb.alpha = 0 if cb.isOver(x, y) else 0.5

                    if clear.isOver(x, y):
                        clear.alpha = 0
                        canvas = np.zeros((720, 1280, 3), np.uint8)
                    else:
                        clear.alpha = 0.5

                if colorsBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    hideColors = not hideColors
                    colorsBtn.text = 'Colors' if hideColors else 'Hide'

                if penBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    hidePenSizes = not hidePenSizes
                    penBtn.text = 'Pen' if hidePenSizes else 'Hide'

                if boardBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    hideBoard = not hideBoard
                    boardBtn.text = 'Board' if hideBoard else 'Hide'

                if pointerBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    usePointer = not usePointer

    # === UI Rendering ===
    for btn in [colorsBtn, boardBtn, penBtn, pointerBtn]:
        btn.drawRect(frame)
        cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), (255, 255, 255), 2)

    if not hideBoard:
        whiteBoard.drawRect(frame)
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)
        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

    penBtn.color = color
    cv2.imshow('Virtual Whiteboard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
