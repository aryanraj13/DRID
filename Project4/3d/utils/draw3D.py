import open3d as o3d
import numpy as np

class Draw3D:
    def __init__(self):
        self.points = []
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()

    def add_point(self, point):
        self.points.append(point)
        self.pcd.points = o3d.utility.Vector3dVector(np.array(self.points))
        self.vis.clear_geometries()
        self.vis.add_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
