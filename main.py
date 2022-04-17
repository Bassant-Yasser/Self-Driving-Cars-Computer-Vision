import cv2
import matplotlib.image as mpimg
import numpy as np
import pickle


def calibrate_camera(images, nx=9, ny=6):
    """"
    images = paths to the images used to calibrate the camera
    nx = Number of inside corners in x
    ny = Number of inside corners in y
    """

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)   
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Apply findChessboardCorners to the individual images
    for fname in images:    
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    # Camera calibration, given object points, image points, and the shape of the grayscale image
    if (len(objpoints) > 0):
        # Camera successfully calibrated.
        print("Camera successfully calibrated.")
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    else:
        # Unable to calibrate the camera.
        print("Unable to calibrate the camera.")
        ret, mtx, dist, rvecs, tvecs = (None, None, None, None, None)

    if not ret:
            raise Exception("Unable to calibrate camera")
    
    # Save camera matrix and distortion coefficient to pickle file
    calibration_param = {}
    calibration_param['mtx'] = mtx
    calibration_param['dist'] = dist
    pickle.dump( calibration_param, open('calibration_pickle.pickle', 'wb') )