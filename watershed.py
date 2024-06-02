# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:11:03 2022

@author: Xinhao Lan
"""
# built-in modules

# third-party modules
import cv2
import numpy as np

def Watershed (path):
    """
    Function for the watershed algorithm.

    :param path: string. Path for the input image.
    
    :return img: np.ndarray. Output images after watershed algorithm.
    """
    # Step0. Read the image and do the normalization
    img = cv2.imread(path)
    cv2.normalize(img, img, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U)
    # Step1. Convert the image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step2. Segmentation with the use of threshold
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Step3. Perform "open operation" on the image, first erode and then dilate
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Step4. Dilate the result of the "open operation" to get areas that are mostly background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Step5. Get the foreground area through distanceTransform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C can only correspond to a mask of 3 DIST_L2 can be 3 or 5
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    # Step6. Subtract sure_bg and sure_fg to get the overlapping area with both foreground and background
    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)
    # Step7. Connected area processing
    ret, markers = cv2.connectedComponents(sure_fg,connectivity=8) 
    markers = markers + 1           
    markers[unknow==255] = 0   
    # Step8. Watershed algorithm
    markers = cv2.watershed(img, markers)  # After the watershed algorithm, the pixels of all contours are marked as -1
    img[markers == -1] = [0, 0, 255]   # Pixels marked with -1 are marked red on the image
    return img

        
def getDiagCycleMat(alpha, beta, n):
    """
    Calculate the 5-diagonal circulant matrix. Used in the snake function.

    :param alpha: int. Input for the alpha in snake function.
    :param beta: int. Input for the beta in snake function.
    :param n: int. Rank for the diagonal matrix.

    :return diag_mat_a + diag_mat_b + diag_mat_c: numpy.ndarray. The 5-diagonal circulant matrix.
    """
    a = 2 * alpha + 6 * beta
    b = -(alpha + 4 * beta)
    c = beta
    diag_mat_a = a * np.eye(n)
    diag_mat_b = b * np.roll(np.eye(n), 1, 0) + b * np.roll(np.eye(n), -1, 0)
    diag_mat_c = c * np.roll(np.eye(n), 2, 0) + c * np.roll(np.eye(n), -2, 0)
    return diag_mat_a + diag_mat_b + diag_mat_c

def getCircleContour(centre=(0, 0), radius=(1, 1), N=200):
    """
    Iterate in the form of a parametric equation to obtain a circle/ellipse contour surrounded by n discrete points.

    :param centre: tuple, optional. Coordinate of the center point. The default is (0, 0).
    :param radius: tuple, optional. The radius of the circle/ellipse. The default is (1, 1).
    :param N: int, optional. The number of discrete points. The default is 200.

    :return np.array([x, y]): numpy.ndarray. 2xN matrix of discrete point.
    """
    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([x, y])


def getRectContour(pt1=(0, 0), pt2=(50, 50)):
    """
    Calculate the initial rectangle contour according to the upper left and lower right point.
    
    Since the Snake model is suitable for smooth curves, this function is not used here.

    :param pt1: tuple, optional. The coordinate for the upper left point. The default is (0, 0).
    :param pt2: tuple, optional. The coordinate for the lower right point. The default is (50, 50).

    :return np.array([x, y]): 2xN matrix of discrete point.
    """
    pt1, pt2 = np.array(pt1), np.array(pt2)
    r1, c1, r2, c2 = pt1[0], pt1[1], pt2[0], pt2[1]
    a, b = r2 - r1, c2 - c1
    length = (a + b) * 2 + 1
    x = np.ones((length), np.float)
    x[:b] = r1
    x[b:a + b] = np.arange(r1, r2)
    x[a + b:a + b + b] = r2
    x[a + b + b:] = np.arange(r2, r1 - 1, -1)
    y = np.ones((length), np.float)
    y[:b] = np.arange(c1, c2)
    y[b:a + b] = c2
    y[a + b:a + b + b] = np.arange(c2, c1, -1)
    y[a + b + b:] = c1
    return np.array([x, y])

def getGaussianPE(src):
    """
    Calculate the Negative Gaussian Potential Energy (NGPE).

    :param src: numpy.ndarray. Input grey image.

    :return E: float. The value of NGPE.
    """
    imblur = cv2.GaussianBlur(src, ksize=(5, 5), sigmaX=3)
    dx = cv2.Sobel(imblur, cv2.CV_16S, 1, 0)  
    dy = cv2.Sobel(imblur, cv2.CV_16S, 0, 1)
    E = dx**2 + dy**2
    return E
 
def snake(img, snake, alpha=0.5, beta=0.1, gamma=0.1, max_iter=2500, convergence=0.01):
    """
    Iterate the Snake model.

    :param img: numpy.ndarray. Input matrix of the image.
    :param snake: numpy.ndarray. 2xN matrix of discrete point of the contour.
    :param alpha: float, optional. Coefficient of elasticity. The default is 0.5.
    :param beta: float, optional. Stiffness factor. The default is 0.1.
    :param gamma: float, optional. Iteration step size. The default is 0.1.
    :param max_iter: int, optional. The maximum number of iterations. The default is 2500.
    :param convergence: float, optional. Convergence threshold. The default is 0.01.

    :return x: numpy.ndarray. X coordinate of points on the final contour.
    :return y: numpy.ndarray. Y coordinate of points on the final contour.
    :return errs: list. List of error value.
    """
    x, y, errs = snake[0].copy(), snake[1].copy(), []
    n = len(x)
    A = getDiagCycleMat(alpha, beta, n)
    inv = np.linalg.inv(A + gamma * np.eye(n))
    y_max, x_max = img.shape
    E_ext = -getGaussianPE(img)
    fx = cv2.Sobel(E_ext, cv2.CV_16S, 1, 0)
    fy = cv2.Sobel(E_ext, cv2.CV_16S, 0, 1)
    T = np.max([abs(fx), abs(fy)])
    fx, fy = fx / T, fy / T
    for g in range(max_iter):
        x_pre, y_pre = x.copy(), y.copy()
        i, j = np.uint8(y), np.uint8(x)
        try:
            xn = inv @ (gamma * x + fx[i, j])
            yn = inv @ (gamma * y + fy[i, j])
        except Exception as e:
            print("Index out of range")
        # reach convergence
        x, y = xn, yn
        err = np.mean(0.5 * np.abs(x_pre - x) + 0.5 * np.abs(y_pre - y))
        errs.append(err)
        if err < convergence:
            print(f"Snake iterates for {g} times and it reasches to convergence.\t err = {err:.3f}")
            break
    return x, y, errs


