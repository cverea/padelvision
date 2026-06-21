import cv2
import numpy as np
import os
class stereo_calibration:
    def __init__(self, folder_left, folder_right):
        self.folder_left = folder_left
        self.folder_right = folder_right
        self.chessboard_size = (7, 7)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []

    def find_corners(self):
        images_left = sorted([os.path.join(self.folder_left, f) for f in os.listdir(self.folder_left) if f.endswith('.jpg')])
        images_right = sorted([os.path.join(self.folder_right, f) for f in os.listdir(self.folder_right) if f.endswith('.jpg')])
        assert len(images_left) == len(images_right), "The number of images in camera1 and camera2 folders must match."

        for img_left_path, img_right_path in zip(images_left, images_right):
            img_left = cv2.imread(img_left_path)
            img_right = cv2.imread(img_right_path)
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)

            if ret_left and ret_right:
                self.objpoints.append(self.objp)
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)
                cv2.drawChessboardCorners(img_left, self.chessboard_size, corners_left, ret_left)
                cv2.drawChessboardCorners(img_right, self.chessboard_size, corners_right, ret_right)
                cv2.imshow('Left Chessboard', img_left)
                cv2.imshow('Right Chessboard', img_right)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

    def calibrate_cameras(self):
        gray_left = cv2.cvtColor(cv2.imread(sorted([os.path.join(self.folder_left, f) for f in os.listdir(self.folder_left) if f.endswith('.jpg')])[0]), cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(cv2.imread(sorted([os.path.join(self.folder_right, f) for f in os.listdir(self.folder_right) if f.endswith('.jpg')])[0]), cv2.COLOR_BGR2GRAY)
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(self.objpoints, self.imgpoints_left, gray_left.shape[::-1], None, None)
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(self.objpoints, self.imgpoints_right, gray_right.shape[::-1], None, None)
        flags = cv2.CALIB_FIX_INTRINSIC
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            mtx_left, dist_left,
            mtx_right, dist_right,
            gray_left.shape[::-1],
            criteria=self.criteria,
            flags=flags
        )
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            cameraMatrix1, distCoeffs1,
            cameraMatrix2, distCoeffs2,
            gray_left.shape[::-1], R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1)
        fs = cv2.FileStorage('stereo_params.yml', cv2.FILE_STORAGE_WRITE)
        fs.write("K1", mtx_left)
        fs.write("D1", dist_left)
        fs.write("K2", mtx_right)
        fs.write("D2", dist_right)
        fs.write("R", R)
        fs.write("T", T)
        fs.write("Q", Q)
        fs.release()
        return map1x, map1y, map2x, map2y, Q

    def compute_disparity(self, map1x, map1y, map2x, map2y, Q):
        img_left = cv2.imread('test_left.jpg')
        img_right = cv2.imread('test_right.jpg')
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        left_rectified = cv2.remap(gray_left, map1x, map1y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(gray_right, map2x, map2y, cv2.INTER_LINEAR)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(left_rectified, right_rectified)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity = np.uint8(disparity)
        cv2.imshow('Disparity Map', disparity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        points_3D = cv2.reprojectImageTo3D(disparity, Q)
        print("3D Points:", points_3D)

        img_left = cv2.imread('test_left.jpg')
        img_right = cv2.imread('test_right.jpg')
        rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

        # Stack images horizontally for comparison
        stacked_images = np.hstack((rectified_left, rectified_right))
        cv2.imshow('Rectified Images', stacked_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def main(self):
        self.find_corners()
        map1x, map1y, map2x, map2y, Q = self.calibrate_cameras()
        self.compute_disparity(map1x, map1y, map2x, map2y, Q)

if __name__ == "__main__":
    folder_left = 'camera1/'
    folder_right = 'camera2/'
    stereo_calib = stereo_calibration(folder_left, folder_right)
    stereo_calib.main()
