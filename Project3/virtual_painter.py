import cv2
import mediapipe as mp
import numpy as np
import time
import os
from math import hypot
from collections import deque

# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpdraw = mp.solutions.drawing_utils

# Header Images
folder = 'colors'
mylist = sorted(os.listdir(folder), reverse=True)
overlist = [cv2.imread(f'{folder}/{i}') for i in mylist]
header = overlist[7]  # Default header
col = (0, 0, 255)

# Canvas & history
canvas = np.zeros((480, 640, 3), np.uint8)
draw_history = []
points_buffer = deque(maxlen=5)

# Drawing vars
xp, yp = 0, 0
sx, sy = 0, 0
brush_thickness = 25
save_message_time = 0
gesture_timer = 0
cooldown = 1.0  # seconds

# Utils
def is_palm_open(lm):
    return all(lm[i][2] < lm[i - 2][2] for i in [8, 12, 16, 20])

def is_undo_gesture(lm):
    return (lm[16][2] < lm[14][2] and  # ring up
            lm[8][2] > lm[6][2] and
            lm[12][2] > lm[10][2] and
            lm[20][2] > lm[18][2])

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    lanmark = []

    if results.multi_hand_landmarks:
        for hn in results.multi_hand_landmarks:
            for id, lm in enumerate(hn.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lanmark.append([id, cx, cy])
            mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)

    if lanmark:
        x1, y1 = lanmark[8][1], lanmark[8][2]
        x2, y2 = lanmark[12][1], lanmark[12][2]
        tx, ty = lanmark[4][1], lanmark[4][2]  # Thumb

        # Smooth fingertip position
        if sx == 0 and sy == 0:
            sx, sy = x1, y1
        else:
            alpha = 0.3
            sx = int(alpha * x1 + (1 - alpha) * sx)
            sy = int(alpha * y1 + (1 - alpha) * sy)

        # Gesture: Clear canvas on open palm
        if is_palm_open(lanmark) and time.time() - gesture_timer > cooldown:
            canvas = np.zeros((480, 640, 3), np.uint8)
            draw_history.clear()
            cv2.putText(frame, 'Canvas Cleared', (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            gesture_timer = time.time()

        # Gesture: Undo
        elif is_undo_gesture(lanmark) and time.time() - gesture_timer > cooldown:
            if draw_history:
                draw_history.pop()
            gesture_timer = time.time()

        # Brush size adjust
        brush_thickness = int(hypot(x1 - tx, y1 - ty) / 2)
        brush_thickness = max(5, min(brush_thickness, 100))

        # Color Picker Mode: Two fingers up
        if lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
            xp, yp = 0, 0
            clicked_color = False
            if sy < 100:
                # Color blocks
                ranges = [
                    (0, (0, 0, 0)),        # Black
                    (80, (226, 43, 138)),  # Pink
                    (160, (255, 255, 255)),# White
                    (240, (0, 255, 0)),    # Green
                    (320, (0, 255, 255)),  # Yellow
                    (400, (255, 0, 0)),    # Blue
                    (480, (0, 0, 255))     # Red
                ]

                for i, (rx, clr) in enumerate(ranges):
                    if rx < sx < rx + 80:
                        header = overlist[i]
                        col = clr
                        clicked_color = True
                        break

                # Save block on right
                if not clicked_color and 560 < sx < 640:
                    if time.time() - gesture_timer > cooldown:
                        os.makedirs('saved_drawings', exist_ok=True)
                        filename = f'saved_drawings/drawing_{int(time.time())}.png'
                        cv2.imwrite(filename, canvas)
                        print(f'[✔] Drawing saved as {filename}')
                        save_message_time = time.time()
                        gesture_timer = time.time()

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

        # Drawing Mode: Index up only
        elif lanmark[8][2] < lanmark[6][2]:
            points_buffer.append((sx, sy))
            if len(points_buffer) >= 3:
                pts = np.array(points_buffer, np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], False, col, brush_thickness)
                draw_history.append(((pts.copy(), col, brush_thickness)))
        else:
            points_buffer.clear()

    # Redraw canvas from history
    redraw = np.zeros((480, 640, 3), np.uint8)
    for pts, color, thick in draw_history:
        cv2.polylines(redraw, [pts], False, color, thick)
    canvas = redraw

    # Final overlay
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    # Header + Save block label
    frame[0:100, 0:640] = header
    cv2.putText(frame, 'Save', (570, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.rectangle(frame, (560, 0), (640, 100), (180, 180, 180), 2)

    # Brush preview
    cv2.circle(frame, (sx, sy), 10, col, -1)
    cv2.putText(frame, f'{brush_thickness}', (sx - 20, sy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Saved message
    if time.time() - save_message_time < 2:
        cv2.putText(frame, "Saved!", (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # FPS
    fps = int(1 / (time.time() - gesture_timer + 0.001))
    cv2.putText(frame, f'FPS: {fps}', (480, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('cam', frame)
    cv2.imshow('canvas', canvas)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('z') and draw_history:
        draw_history.pop()

cap.release()
cv2.destroyAllWindows()
