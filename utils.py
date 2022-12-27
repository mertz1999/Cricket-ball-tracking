import cv2 
import numpy as np  


def draw_points(img, points):
    for point in points:
        cv2.circle(img, point, 1, (0,0,0), 3)