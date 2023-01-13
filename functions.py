import numpy as np
import math

def cam_to_wrld(cam, cam_coord):
    extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
    extrinsics = np.linalg.inv(extrinsics)
    homo_cam_coord = np.array([[cam_coord[0]], [cam_coord[1]], [cam_coord[2]], [1]])
    wrld_coord = np.matmul(extrinsics, homo_cam_coord).astype("double")

    return np.array([wrld_coord[0][0], wrld_coord[1][0], wrld_coord[2][0]])

# transform world space to camera space
def wrld_to_cam(cam, wrld_coord):
    extrinsics = np.concatenate((cam.extrinsics, [np.array([0, 0, 0, 1])]), axis=0)
    extrinsics = extrinsics
    homo_wrld_coord = np.array([[wrld_coord[0]], [wrld_coord[1]], [wrld_coord[2]], [1]])
    cam_coord = np.matmul(extrinsics, homo_wrld_coord).astype("double")

    return np.array([cam_coord[0], cam_coord[1], cam_coord[2]])

def calc_vec_angle(vec1, vec2):
    return math.acos(np.dot(vec1, vec2) / np.linalg.norm(vec1)*np.linalg.norm(vec2)) / math.pi * 180