import imageio
import cv2
import random
import numpy as np
# import matplotlib.pyplot as plt
from itertools import starmap

def endpoints(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x_0 = a * rho
    y_0 = b * rho
    x_1 = int(x_0 + 1000 * (-b))
    y_1 = int(y_0 + 1000 * (a))
    x_2 = int(x_0 - 1000 * (-b))
    y_2 = int(y_0 - 1000 * (a))
    return ((x_1, y_1), (x_2, y_2))


def sample_lines(lines, size):
    if size > len(lines):
        size = len(lines)
    return random.sample(lines, size)


def line_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((15, 15), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    edges = cv2.Canny(opening, 40, 150, apertureSize=3)  # Canny edge detection

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # Hough line detection
    hough_lines = []
    if lines is not None:
        for line in lines:
            hough_lines.extend(list(starmap(endpoints, line)))

    sampled_lines = sample_lines(hough_lines, 20)

    line_det = np.zeros(img.shape[:2])
    for line in sampled_lines:
        cv2.line(line_det, line[0], line[1], (255), 2)

    return edges[None,:,:], line_det[None,:,:]