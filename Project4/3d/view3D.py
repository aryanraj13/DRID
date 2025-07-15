import open3d as o3d
import sys

def view_3d_drawing(file_path):
    print(f"Opening 3D drawing: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd or len(pcd.points) == 0:
        print("No points found or invalid file.")
        return

    # Optionally estimate normals (if you want)
    pcd.estimate_normals()

    # Visualize
    o3d.visualization.draw_geometries([pcd],
        window_name="3D Drawing Viewer",
        width=800,
        height=600,
        point_show_normal=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view3D.py <drawing_file.ply>")
    else:
        view_3d_drawing(sys.argv[1])
