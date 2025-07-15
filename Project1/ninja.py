import cv2
import time
import random
import mediapipe as mp # type: ignore
import math
import numpy as np

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Game Variables
curr_Frame = 0
prev_Frame = 0
delta_time = 0

next_Time_to_Spawn = 0
Speed = [0, 5]
Fruit_Size = 30
Spawn_Rate = 1
Score = 0
Lives = 15
Difficulty_level = 1
game_Over = False
last_level_up_score = 0

slash = np.empty((0, 2), np.int32)
slash_Color = (255, 255, 255)
slash_length = 19

Fruits = []

# ===================== Helper Functions ======================

def Spawn_Fruits():
    fruit = {}
    random_x = random.randint(15, 600)
    random_color = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )
    fruit["Color"] = random_color
    fruit["Curr_position"] = [random_x, 440]
    fruit["Next_position"] = [0, 0]
    Fruits.append(fruit)

def Fruit_Movement(fruits, speed):
    global Lives

    # Move fruits and draw them
    updated_fruits = []
    for fruit in fruits:
        if fruit["Curr_position"][1] < 20 or fruit["Curr_position"][0] > 650:
            Lives -= 1
            continue  # skip adding this fruit (i.e., remove it)

        cv2.circle(img, tuple(fruit["Curr_position"]), Fruit_Size, fruit["Color"], -1)
        fruit["Next_position"][0] = fruit["Curr_position"][0] + speed[0]
        fruit["Next_position"][1] = fruit["Curr_position"][1] - speed[1]
        fruit["Curr_position"] = fruit["Next_position"]
        updated_fruits.append(fruit)

    fruits[:] = updated_fruits

def distance(a, b):
    return int(math.hypot(a[0] - b[0], a[1] - b[1]))

# ===================== Main Game Loop ======================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Skipping frame...")
        continue

    h, w, c = img.shape
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect Hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 8:  # Index finger tip
                    index_pos = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(img, index_pos, 18, slash_Color, -1)

                    # Add point to slash trail
                    slash = np.append(slash, [index_pos], axis=0)
                    while len(slash) > slash_length:
                        slash = np.delete(slash, 0, axis=0)

                    # Check collisions
                    remaining_fruits = []
                    for fruit in Fruits:
                        d = distance(index_pos, fruit["Curr_position"])
                        if d < Fruit_Size:
                            Score += 100
                            slash_Color = fruit["Color"]
                            continue  # sliced, remove fruit
                        remaining_fruits.append(fruit)
                    Fruits = remaining_fruits

    # Difficulty Scaling
    if Score - last_level_up_score >= 1000 and Score != 0:
        Difficulty_level += 1
        last_level_up_score = Score
        Spawn_Rate = Difficulty_level * 4 / 5
        Speed[0] = Speed[0] * Difficulty_level
        Speed[1] = int(5 * Difficulty_level / 2)

    if Lives <= 0:
        game_Over = True

    # Draw slash
    if len(slash) >= 2:
        cv2.polylines(img, [slash.reshape((-1, 1, 2))], False, slash_Color, 15)

    # FPS
    curr_Frame = time.time()
    delta_time = curr_Frame - prev_Frame
    FPS = int(1 / delta_time) if delta_time > 0 else 0
    prev_Frame = curr_Frame

    # Game Texts
    cv2.putText(img, "FPS : " + str(FPS), (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
    cv2.putText(img, "Score: " + str(Score), (int(w * 0.35), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
    cv2.putText(img, "Level: " + str(Difficulty_level), (int(w * 0.01), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
    cv2.putText(img, "Lives remaining : " + str(Lives), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Game Logic
    if not game_Over:
        if time.time() > next_Time_to_Spawn:
            Spawn_Fruits()
            next_Time_to_Spawn = time.time() + (1 / Spawn_Rate)
        Fruit_Movement(Fruits, Speed)
    else:
        cv2.putText(img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        Fruits.clear()

    cv2.imshow("Ninja Fruit Game", img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Cleanup
hands.close()
cap.release()
cv2.destroyAllWindows()
