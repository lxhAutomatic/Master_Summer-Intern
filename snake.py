# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:06:55 2022

@author: Xinhao Lan
"""

import numpy as np
from skimage.io import imread
import cv2
from matplotlib import pyplot as plt

def getDiagCycleMat(alpha, beta, n):
    """
    Calculate the 5-diagonal circulant matrix. Used in the snake function.

    Parameters
    ----------
    alpha : int
        Input for the alpha in snake function
    beta : int
        Input for the beta in snake function
    n : int
        Rank for the diagonal matrix

    Returns
    -------
    list
        The 5-diagonal circulant matrix

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

    Parameters
    ----------
    centre : tuple, optional
        Coordinate of the center point. The default is (0, 0).
    radius : tuple, optional
        The radius of the circle/ellipse. The default is (1, 1).
    N : int, optional
        The number of discrete points. The default is 200.

    Returns
    -------
    list
        2xN matrix of discrete point

    """
    t = np.linspace(0, 2 * np.pi, N)
    x = centre[0] + radius[0] * np.cos(t)
    y = centre[1] + radius[1] * np.sin(t)
    return np.array([x, y])


def getRectContour(pt1=(0, 0), pt2=(50, 50)):
    """
    Calculate the initial rectangle contour according to the upper left and lower right point
    Since the Snake model is suitable for smooth curves, this function is not used here
    
    Parameters
    ----------
    pt1 : tuple, optional
        The coordinate for the upper left point. The default is (0, 0).
    pt2 : TYPE, optional
        The coordinate for the lower right point. The default is (50, 50).

    Returns
    -------
    list
        2xN matrix of discrete point

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
    Calculate the Negative Gaussian Potential Energy (NGPE)

    Parameters
    ----------
    src : list
        Input grey image

    Returns
    -------
    E : float
        the value of NGPE

    """
    imblur = cv2.GaussianBlur(src, ksize=(5, 5), sigmaX=3)
    dx = cv2.Sobel(imblur, cv2.CV_16S, 1, 0)  
    dy = cv2.Sobel(imblur, cv2.CV_16S, 0, 1)
    E = dx**2 + dy**2
    return E
 
def snake(img, snake, alpha=0.5, beta=0.1, gamma=0.1, max_iter=2500, convergence=0.01):
    """
    Iterate the Snake model

    Parameters
    ----------
    img : list
        Input image
    snake : list
        2xN matrix of discrete point of the contour
    alpha : float, optional
        Coefficient of elasticity. The default is 0.5.
    beta : float, optional
        Stiffness factor. The default is 0.1.
    gamma : float, optional
        Iteration step size. The default is 0.1.
    max_iter : int, optional
        The maximum number of iterations. The default is 2500.
    convergence : float, optional
        Convergence threshold. The default is 0.01.

    Returns
    -------
    x : list
        X coordinate of points on the final contour.
    y : list
        Y coordinate of points on the final contour.
    errs : list
        List of error value.

    """
    x, y, errs = snake[0].copy(), snake[1].copy(), []
    n = len(x)
    A = getDiagCycleMat(alpha, beta, n)
    inv = np.linalg.inv(A + gamma * np.eye(n))
    y_max, x_max = img.shape
    max_px_move = 1.0
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

img = imread('D://SteelImage//test//SIS_00018095_10_0864.jpg')
cv2.normalize(img, img, 0, 256, cv2.NORM_MINMAX, cv2.CV_8U)
init = getCircleContour((img.shape[1]/2, img.shape[0]/2), (img.shape[1]/2-1, img.shape[0]/2-1), N=200)
#init = getCircleContour((78, 128), (9, 128), N=200)
x, y, errs = snake(img, snake=init, alpha=0.1, beta=1, gamma=0.1)

plt.figure() 
plt.imshow(img, cmap="gray")
plt.plot(init[0], init[1], '--r', lw=1)
plt.plot(x, y, 'g', lw=1)
plt.xticks([]), plt.yticks([]), plt.axis("off")
plt.figure()
plt.plot(range(len(errs)), errs)
plt.show()

