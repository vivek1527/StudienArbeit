import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import cv2
from skimage.feature import match_descriptors
import os
import Studienarbeit_corregistration.caps_implementation.config as config
from Studienarbeit_corregistration.caps_implementation.CAPS.caps_model import CAPSNet
from Studienarbeit_corregistration.dense_correspondense_matcher import extract_superpoint_keypoints, extract_CAPSDescriptor, nearestneighboursMatching





def mma(pts1, pts2,  matches, threshold):
    """
    Finds the no. of repeated points in both images given the keypoints
    param: pts1- (keypoints found in 2nd image) listed in numpy array of form [(y1, x1), (y2, x2)...]
    param: pts2 - (keypoints in 2nd image found using groundtruth homography) listed in numpy array of form [(y1, x1), (y2, x2)...]
    param: correct_distance - distance used to approximate whether a points is nearby or not
    """

    N1 = pts1.shape[0]
    print("The total number of matches are: ", N1)
    pts1 = pts1[np.newaxis, ...]
    pts2 = pts2[:, np.newaxis, :]
    dist = np.linalg.norm(pts1 - pts2, axis=2)
    if N1 != 0:
        min_dist = np.min(dist, axis=1)
        count = np.sum(min_dist <= threshold)

    mma_metric = (count) / matches
    return mma_metric


if __name__ == '__main__':
    mma_pixels_order = []
    for i in range(1, 2):
        mma_avg = []

        file_directory = '/media/vivekramayanam/STUDY/CSE/4-sem/CoarseFeatureMatching/meanmatchingaccuracy/' # mention your file directory
        for file in os.listdir(file_directory):
            (width, height) = (865, 579)


            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            image1 = cv2.imread(os.path.join(file_directory, file))
            image1 = cv2.resize(image1, (width, height), interpolation=cv2.INTER_AREA)
            keypoint1, descriptor1 = extract_superpoint_keypoints(image1)
            desc1 = extract_CAPSDescriptor(keypoint1, image1)
            #homograpymat = sample_homography(np.array([579, 865]), perspective=True, scaling=True, rotation=True,
            #                                 translation=True)

            homograpymat = np.array([[0.68, 0.577, -118.76], [-0.349, 0.741, 185.75], [-0.00, -0.000, 1]])
            image1_transformed = cv2.warpPerspective(image1, homograpymat, (image1.shape[1], image1.shape[0]))
            image_save = cv2.cvtColor(image1_transformed, cv2.COLOR_BGR2RGB)
            #plt.imshow(image_save)
            #plt.show()
            keypoint2, descriptor2 = extract_superpoint_keypoints(image1_transformed)
            desc2 = extract_CAPSDescriptor(keypoint2, image1_transformed)
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(image1)
            axes[1].imshow(image1_transformed)
            # plt.show()
            args = config.get_args()
            args.batch_size = 1
            args.phase = "test"
            coarse_correspondence_model = CAPSNet(args, device)
            matches = match_descriptors(desc1, desc2, cross_check=True, max_ratio=0.95)

            keypoint1 = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoint1])
            keypoints_left = keypoint1[matches[:, 0], : 2]

            keypoint2 = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoint2])
            keypoints_right = keypoint2[matches[:, 1], : 2]

            keypoints_left = keypoints_left.reshape(-1, 1, 2)
            keypoints_right_homography = cv2.perspectiveTransform(keypoints_left, homograpymat)

            keypoints_right_homography = np.squeeze(keypoints_right_homography, axis=1)

            mma_score = mma(keypoints_right_homography, keypoints_right, len(matches), i)

            mma_avg.append(mma_score)
        mma_pixels_order.append(np.mean(mma_avg))
    print(mma_pixels_order)



