# main3D.py
from drawing_utils import load_3d_drawing, show_3d_drawing

def main():
    path = 'drawings/my_drawing.csv'
    print(f"Loading 3D drawing from: {path}")
    points = load_3d_drawing(path)
    show_3d_drawing(points)

if __name__ == '__main__':
    main()
