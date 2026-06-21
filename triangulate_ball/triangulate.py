import numpy as np
import cv2
import pandas as pd
class Triangulation:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.camera1_data = None
        self.camera2_data = None
        self.points_3D_array = None

    def generate_mock_data(self):
        x1 = np.linspace(0.2, 0.4, self.num_frames) + np.random.normal(0, 0.01, self.num_frames)
        y1 = np.linspace(3.2, 3.6, self.num_frames) + np.random.normal(0, 0.01, self.num_frames)
        x2 = np.linspace(0.3, 0.5, self.num_frames) + np.random.normal(0, 0.01, self.num_frames)
        y2 = np.linspace(3.3, 3.7, self.num_frames) + np.random.normal(0, 0.01, self.num_frames)

        self.camera1_data = pd.DataFrame({'x1': x1, 'y1': y1})
        self.camera2_data = pd.DataFrame({'x2': x2, 'y2': y2})

    def triangulate(self):
        points_2D_cam1 = self.camera1_data[['x1', 'y1']].values
        points_2D_cam2 = self.camera2_data[['x2', 'y2']].values

        K1 = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
        K2 = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])

        R1 = np.array([[-1, 0, 0], [0, 0.9063, -0.4226], [0, -0.4226, -0.9063]])
        R2 = np.array([[1, 0, 0], [0, 0.9063, -0.4226], [0, 0.4226, 0.9063]])

        T1 = np.array([[-26.885], [0], [10]])
        T2 = np.array([[26.885], [0], [10]])

        P1 = np.dot(K1, np.hstack((R1, T1)))
        P2 = np.dot(K2, np.hstack((R2, T2)))

        points_2D_cam1_hom = np.vstack([points_2D_cam1[:, 0], points_2D_cam1[:, 1]])
        points_2D_cam2_hom = np.vstack([points_2D_cam2[:, 0], points_2D_cam2[:, 1]])

        points_4D_hom = cv2.triangulatePoints(P1, P2, points_2D_cam1_hom, points_2D_cam2_hom)
        self.points_3D_array = (points_4D_hom[:3, :] / points_4D_hom[3, :]).T

    def main(self):
        self.generate_mock_data()
        self.triangulate()
        return self.points_3D_array

if __name__ == "__main__":
    triangulation = Triangulation(num_frames=10)
    points_3D_array = triangulation.main()
    print(points_3D_array)
