import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
from drawing_utils import save_3d_drawing

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# OpenCV
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Open3D
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("3D Drawing", width=800, height=600)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Set a fixed view point
view_ctl = vis.get_view_control()
view_ctl.set_zoom(0.7)

# Track points
points3D = []
smooth_points = []

def is_only_index_up(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x)
    # Others
    for i in range(1, 5):
        fingers.append(landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y)

    return fingers[1] and not any(fingers[i] for i in [0, 2, 3, 4])

def update_point_cloud():
    if not points3D:
        return
    xyz = np.array(points3D, dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Optional: draw connecting lines for visibility
    if len(points3D) > 1:
        lines = [[i, i+1] for i in range(len(points3D)-1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = pcd.points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])  # Blue lines
        vis.clear_geometries()
        vis.add_geometry(line_set)
    else:
        vis.clear_geometries()
        vis.add_geometry(pcd)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

def save_callback(vis):
    save_3d_drawing(points3D)
    print("✅ Saved 3D drawing to drawings/my_drawing.csv")
    return False

def quit_callback(vis):
    print("👋 Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    exit(0)

# Register key callbacks
vis.register_key_callback(ord("s"), save_callback)
vis.register_key_callback(ord("q"), quit_callback)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        lm = hand_lms.landmark[8]
        x, y, z = int(lm.x * w), int(lm.y * h), lm.z

        if is_only_index_up(hand_lms.landmark):
            smooth_points.append((x, y, z))
            if len(smooth_points) > 5:
                smooth_points.pop(0)

            avg_x = int(np.mean([pt[0] for pt in smooth_points]))
            avg_y = int(np.mean([pt[1] for pt in smooth_points]))
            avg_z = np.mean([pt[2] for pt in smooth_points])

            # Fix z-scale and coordinate space
            norm_x = avg_x
            norm_y = avg_y
            norm_z = -avg_z * 1000  # Flip and scale

            points3D.append((norm_x, norm_y, norm_z))
            update_point_cloud()

    # Webcam UI
    cv2.imshow("Camera Feed", frame)
    vis.poll_events()
    vis.update_renderer()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
