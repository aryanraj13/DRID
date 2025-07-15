# drawing_utils.py
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_3d_drawing(points3D, filename='drawings/my_drawing.csv'):
    if not points3D:
        print("No 3D points to save.")
        return
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z'])
        writer.writerows(points3D)
    print(f"Saved 3D drawing with {len(points3D)} points.")

def load_3d_drawing(filename='drawings/my_drawing.csv'):
    points3D = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                points3D.append((x, y, z))
    except FileNotFoundError:
        print(f"File not found: {filename}")
    return points3D

def show_3d_drawing(points3D):
    if not points3D:
        print("No 3D points to show.")
        return
    xs, ys, zs = zip(*points3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, c='blue')
    ax.set_title("3D Drawing")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Depth)")
    plt.show()
