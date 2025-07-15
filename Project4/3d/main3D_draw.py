import cv2
import numpy as np
import open3d as o3d
from collections import deque
import time

# Green color HSV range
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])

# Smoothing buffer
smooth_buffer = deque(maxlen=5)

# 3D drawing data
points_3d = []
lines = []

# Drawing state
draw_enabled = True
erase_mode = False

# Open3D window
vis = o3d.visualization.Visualizer()
vis.create_window()
line_set = o3d.geometry.LineSet()

# Webcam
cap = cv2.VideoCapture(0)

print("Press 'd' to toggle draw mode (ON/OFF)")
print("Hold 'e' to erase")
print("Press 's' to save as PLY")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        draw_enabled = not draw_enabled
        print(f"Draw Mode: {'ON' if draw_enabled else 'OFF'}")
        time.sleep(0.3)  # prevent rapid toggling
    elif key == ord('s'):
        timestamp = int(time.time())
        filename = f"drawing_{timestamp}.csv"
        np.savetxt(filename, np.array(points_3d), delimiter=",", header="x,y,z", comments='')
        print(f"Saved CSV: {filename}")

    elif key == ord('e'):
        erase_mode = True
    else:
        erase_mode = False

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Smoothing
                smooth_buffer.append((cx, cy))
                if len(smooth_buffer) == smooth_buffer.maxlen:
                    avg_x = int(np.mean([p[0] for p in smooth_buffer]))
                    avg_y = int(np.mean([p[1] for p in smooth_buffer]))

                    cv2.circle(frame, (avg_x, avg_y), 8, (0, 255, 0), -1)

                    fx, fy = 100, 100
                    z = 0.5  # Fixed depth
                    pt3d = [avg_x / fx, avg_y / fy, z]

                    if erase_mode:
                        if len(points_3d) >= 10:
                            print("Erasing last stroke...")
                            points_3d = points_3d[:-10]
                            lines = lines[:-9]
                    elif draw_enabled:
                        points_3d.append(pt3d)
                        if len(points_3d) >= 2:
                            lines.append([len(points_3d) - 2, len(points_3d) - 1])

                    # Update Open3D
                    line_set.points = o3d.utility.Vector3dVector(points_3d)
                    line_set.lines = o3d.utility.Vector2iVector(lines)

                    vis.clear_geometries()
                    vis.add_geometry(line_set)
                    vis.poll_events()
                    vis.update_renderer()

    cv2.putText(frame, f"Draw: {'ON' if draw_enabled else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if draw_enabled else (0, 0, 255), 2)

    cv2.imshow("Green Pointer Tracker", frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
