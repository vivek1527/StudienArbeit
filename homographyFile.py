import cv2
import numpy as np
import matplotlib.pyplot as plt


HomographyMat1 = np.asarray([[1, 0.2, 0], [0.2, 1, 0], [0, 0, 1]], dtype=np.float32)
im1 = cv2.imread('/home/vivekramayanam/PycharmProjects/Studienarbeit_corregisteration_new/Studienarbeit_corregistration/HomographyCreation/IMG_1994.JPG')
dim = (640, 480)
im1 = cv2.resize(im1, dim)
im2_dst = im1
im2 = cv2.warpPerspective(im1, HomographyMat1, (im2_dst.shape[1], im2_dst.shape[0]))
cv2.imshow('im1', im1)
cv2.imshow('im2', im2)
cv2.waitKey(0)
