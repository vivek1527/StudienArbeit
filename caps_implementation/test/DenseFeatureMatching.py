import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torch
from pathlib import Path
from cv2 import cv2
import torch.nn.functional as F
import torch.nn as nn
import csv
from numpy.random import normal
from torchvision import transforms
from numpy.random import uniform
from scipy.stats import truncnorm
from torch.autograd import Variable
from skimage.feature import match_descriptors
import os
import torchmetrics
import Studienarbeit_corregistration.caps_implementation.config as config
from Studienarbeit_corregistration.caps_implementation.dataloader.megadepth import MegaDepthLoader
import Studienarbeit_corregistration.caps_implementation.utils as utils
from Studienarbeit_corregistration.caps_implementation.CAPS.caps_model import CAPSModel, CAPSNet
from Studienarbeit_corregistration.dense_correspondense_matcher import extract_SIFT_keypoints, extract_superpoint_keypoints, extract_CAPSDescriptor, nearestneighboursMatching








def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.
    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homographies: batched or not (shapes (B, 3, 3) and (...) respectively).
        device: gpu device or cpu
    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies[np.newaxis,...] if no_batches else homographies
    batch_size = homographies.shape[0]
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

    homographies = homographies.reshape([batch_size * 3, 3])
    warped_points = homographies @ points.T
    warped_points = warped_points.reshape([batch_size, 3, -1])
    warped_points = warped_points.T.squeeze()
    warped_points = warped_points[:, :2] / warped_points[:, 2:]
    return warped_points




def pck(pts1, pts2, threshold=5):
    """
    Finds the no. of repeated points in both images given the keypoints
    param: pts1- (keypoints found in an image) listed in numpy array of form [(y1, x1), (y2, x2)...]
    param: pts2 - (keypoints found using ground truth homography) listed in numpy array of form [(y1, x1), (y2, x2)...]
    param: threshold - distance used to approximate whether a points is nearby or not
    """

    N1 = pts1.shape[0]
    pts1 = pts1[np.newaxis, ...]
    pts2 = pts2[:, np.newaxis, :]
    dist = np.linalg.norm(pts1 - pts2, axis=2)
    if N1 != 0:
        min_dist = np.min(dist, axis=1)
        count = np.sum(min_dist <= threshold)

    pckmetric = count / N1
    return pckmetric


def filter_points(points, shape, indicesTrue=False):
    #  check!
    x_warp, y_warp = points[:, 0].astype(np.int64), points[:, 1].astype(np.int64)
    indices = np.where((x_warp >= 0) & (x_warp < shape[0]) & (y_warp >= 0) & (y_warp < shape[1]))
    points_warp = np.asarray(list(zip(y_warp[indices[0]], x_warp[indices[0]])))
    if indicesTrue:
        return indices
    return points_warp

def sample_homography(shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
                      n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.3,
                      perspective_amplitude_y=0.1, patch_ratio=1, max_angle=np.pi / 8,
                      allow_artifacts=True, translation_overflow=0):
    """Sample a random valid homography.
    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.
    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        shift:
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """
    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]])
    # Random perspective and affine perturbations
    # lower, upper = 0, 2
    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y / 2).rvs(1)
        h_displacement_left = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)
        h_displacement_right = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = truncnorm(-1 * std_trunc, std_trunc, loc=1, scale=scaling_amplitude / 2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx, :, :]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0], 1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis, :, :], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]
    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]
    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    return homography

if __name__ == '__main__':
    # (width, height) = (868, 579)
    (width, height) = (434, 290)
    folder = '/home/vivekramayanam/PycharmProjects/Studienarbeit_corregisteration_new/Studienarbeit_corregistration/Library_test/'
    file_name = os.listdir(folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = cv2.imread(folder + file_name[2])
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    keypoint1, desc1 = extract_superpoint_keypoints(image)
    coord1 = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoint1])
    coord1 = torch.from_numpy(coord1).float()

    homograpymat = sample_homography(np.array([290, 434]), perspective=True, scaling=False, rotation=False, translation=False)
    print(homograpymat)
    image_transformed = cv2.warpPerspective(image, homograpymat, (image.shape[1], image.shape[0]))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(image_transformed)
    plt.show()
    args = config.get_args()
    args.batch_size = 1
    args.ckpth_path = "/Studienarbeit_corregistration/caps-pretrained.pth"
    args.phase = "test"
    densecorrespondence_model = CAPSNet(args, device)

    img_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    image1 = np.copy(image)
    image = torch.from_numpy(image).float().to(device) / 255.0
    image = torch.unsqueeze(image, dim=0)
    image = image.permute(0, 3, 1, 2)
    image_tensor = img_transform(image)
    image_transformed = torch.from_numpy(image_transformed).float().to(device) / 255.0
    image_transformed = torch.unsqueeze(image_transformed, dim=0)
    image_transformed = image_transformed.permute(0, 3, 1, 2)
    image_transformed_tensor = img_transform(image_transformed)

    coord1 = torch.unsqueeze(coord1, dim=0)
    im1 = Variable(image_tensor.to(device))
    im2 = Variable(image_transformed_tensor.to(device))

    coord1_ = Variable(coord1.to(device))
    coord2, std = densecorrespondence_model.test(im1, im2, coord1_)
    coord1 = coord1_.squeeze().cpu().numpy()
    #print(coord1.shape)
    coord2_dense = coord2.squeeze().cpu().numpy()
    #print(coord2_dense)

    dummy_image = np.zeros_like(image1)

    coord2_dense_reversed = np.stack([coord2_dense[:, 1], coord2_dense[:, 0]]).astype(np.int)
    dummy_image[coord2_dense_reversed[0, :], coord2_dense_reversed[1, :]] = [255, 255, 255]
    mask = np.ones_like(image1)
    mask_transformed = cv2.warpPerspective(mask, homograpymat, dsize=image1.shape[:2][::-1])
    new_keypoints = mask_transformed * dummy_image
    coord2dense_prune = np.nonzero(new_keypoints[..., 0])
    coord2dense_prune = np.stack([coord2dense_prune[1], coord2dense_prune[0]]).T
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(mask_transformed * 255)
    axes[1].imshow(new_keypoints)
    plt.show()

    inv_homography = np.linalg.pinv(homograpymat)

    coord2_dense_predicted_img1 = warp_points(coord2dense_prune, inv_homography)
    coord2_dense_predicted_img1 = filter_points(coord2_dense_predicted_img1, (640, 480))
    #print(coord2_dense_predicted_img1)
    metric = pck(coord1, coord2_dense_predicted_img1)
    print(metric)






